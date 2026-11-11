import copy
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import DatasetLoader
from loss import ValueNetwork, TwinNetwork, GaussianPolicy

def expectileLoss(diff, tau):
    weight = torch.where(diff>0, tau, 1-tau)
    return (weight*diff**2).mean()

class IQLAgent:
    def __init__(
            self,
            loader: DatasetLoader,
            lr=3e-4,
            gamma=0.99,
            tau=0.7,
            beta=3.0,
            alpha=0.005,
            max_steps=1_000_000,
            dropout=0.0
    ):
        self.loader = loader
        self.device = loader.device
        self.gamma = gamma
        self.tau = tau
        self.beta = beta
        self.alpha = alpha


        maxAction = float(loader.env.action_space.high[0])


        self.v_net = ValueNetwork(loader.obsDim).to(self.device)
        self.q_net = TwinNetwork(loader.obsDim, loader.actDim).to(self.device)


        self.q_target = copy.deepcopy(self.q_net).to(self.device)
        for param in self.q_target.parameters():
            param.requires_grad = False


        self.policy_net = GaussianPolicy(
            loader.obsDim, loader.actDim, maxAction=maxAction, dropout=dropout
        ).to(self.device)



        self.v_optimizer = torch.optim.Adam(self.v_net.parameters(), lr=lr)
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)


        self.policy_lr_scheduler = CosineAnnealingLR(
            self.policy_optimizer, T_max=max_steps
        )

    def polyakUpdate(self):
        for targetParam, param in zip(self.q_target.parameters(), self.q_net.parameters()):
            targetParam.data.mul_(1.0-self.alpha).add_(param.data, alpha=self.alpha)


    def update_v(self, batchSize=256, batch=None):
        if batch is None:
            batch = self.loader.samplebatch(batchSize)

        obs, act = batch['observations'], batch['actions']

        with torch.no_grad():
            q_targ = self.q_target.qMin(obs, act)

        v = self.v_net(obs)
        u = q_targ - v
        v_loss = expectileLoss(q_targ-v, self.tau)

        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        return v_loss.item(), v.mean().item()

    def update_q(self, batchSize=256, batch=None):
        if batch is None:
            batch = self.loader.samplebatch(batchSize)
        obs, act = batch['observations'], batch['actions']
        next_obs = batch['next_observations']
        rewards, terminations = batch['rewards'], batch['terminations']


        with torch.no_grad():
            next_v = self.v_net(next_obs)
            target_q = rewards + self.gamma*(1.0-terminations)*next_v

        q1, q2 = self.q_net(obs, act)
        q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        self.polyakUpdate()
        meanQ = torch.min(q1, q2).mean().item()
        return q_loss.item(), meanQ

    def update_policy(self, batchSize=256, maxWeight=100.0, batch=None):
        if batch is None:
            batch = self.loader.samplebatch(batchSize)

        obs, act = batch['observations'], batch['actions']

        with torch.no_grad():
            q_targ = self.q_target.qMin(obs, act)
            v = self.v_net(obs)
            adv = q_targ-v
            weights = torch.clamp(torch.exp(self.beta*adv), max=maxWeight)

        logProb = self.policy_net.logProb(obs, act)
        policyLoss = -(weights*logProb).mean()

        self.policy_optimizer.zero_grad()
        policyLoss.backward()
        self.policy_optimizer.step()
        self.policy_lr_scheduler.step()

        return policyLoss.item(), weights.mean().item()
