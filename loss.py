import torch
from torch.distributions import Normal, Independent
import torch.nn as nn


def buildMLP(inputDim, outputDim, hiddenDim=(256,256), dropout=0.0, outputActivation=None):
    layers = []
    for hDim in hiddenDim:
        layers.append(nn.Linear(inputDim, hDim))
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        inputDim = hDim

    layers.append(nn.Linear(inputDim, outputDim))
    if outputActivation is not None:
        layers.append(outputActivation)
    return nn.Sequential(*layers)



class ValueNetwork(nn.Module):
    def __init__(self, obsDim, hiddenDims=(256,256)):
        super().__init__()
        self.net = buildMLP(inputDim=obsDim, outputDim=1, hiddenDim=hiddenDims)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class TwinNetwork(nn.Module):
    def __init__(self, obsDim, actDim, hiddenDims=(256,256)):
        super().__init__()
        self.q1 = buildMLP(obsDim+actDim,1,hiddenDim=hiddenDims)
        self.q2 = buildMLP(obsDim+actDim,1,hiddenDim=hiddenDims)


    def forward(self, obs, act):
        sa = torch.cat([obs, act], dim=-1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)

    def qMin(self, obs, act):
        q1, q2 = self.forward(obs, act)
        return torch.min(q1, q2)


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class GaussianPolicy(nn.Module):
    def __init__(self, obsDim, actDim, maxAction=1.0, hiddenDims=(256,256), dropout=0.0):
        super().__init__()
        self.maxAction = maxAction

        self.meanNet = buildMLP(
            obsDim,
            actDim,
            hiddenDim=hiddenDims,
            dropout=dropout,
            outputActivation=nn.Tanh()
        )

        self.logStd = nn.Parameter(torch.zeros(actDim, dtype=torch.float32))



    def forward(self, obs):
        mean = self.meanNet(obs)*self.maxAction
        logStd = torch.clamp(self.logStd, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(logStd)

        return Independent(Normal(mean, std),1)

    def logProb(self, obs, act):
        dist = self.forward(obs)
        return dist.log_prob(act)


    @torch.no_grad()
    def act(self, obs, deterministic=True):
        dist = self.forward(obs)
        action = dist.mean if deterministic else dist.sample()
        return torch.clamp(action, -self.maxAction, self.maxAction)