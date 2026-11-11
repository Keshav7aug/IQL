from utils import DatasetLoader
from agent import IQLAgent
import torch
import numpy as np
@torch.no_grad()
def evaluate(agent: IQLAgent, nEpisodes=10):
    loader = agent.loader
    evalEnv = loader.evalEnv
    agent.policy_net.eval()
    episodeReturns = []
    for _ in range(nEpisodes):
        obs, _ = evalEnv.reset()
        terminated = False
        truncated = False
        totalReward = 0.0

        while not (truncated or terminated):
            procObs = loader.processObs(obs)
            obsTensor = torch.as_tensor(
                procObs, dtype=torch.float32, device=agent.device
            ).unsqueeze(0)

            action = agent.policy_net.act(obsTensor, deterministic=True)
            actionNP = action.squeeze(0).cpu().numpy()

            obs, reward, terminated, truncated, _ = evalEnv.step(actionNP)
            totalReward += float(reward)
        episodeReturns.append(totalReward)
    agent.policy_net.train()

    meanReturn = float(np.mean(episodeReturns))
    meanScore = float(loader.normalizedScore(meanReturn))

    return meanReturn, meanScore

def train(steps=1_000_000):
    loader = DatasetLoader(device='cuda')
    agent = IQLAgent(
        loader=loader,
        lr=3e-4,
        gamma=0.99,
        tau=0.7 if loader.specialDataset == 'locomotion' else 0.9,
        beta=3.0 if loader.specialDataset == 'locomotion' else 10.0
    )


    for step in range(1,steps+1):
        batch = loader.samplebatch(batchSize=256)


        v_loss, mean_v = agent.update_v(batch=batch)
        pi_loss, mean_adv_wt = agent.update_policy(batch=batch)
        q_loss, mean_q = agent.update_q(batch=batch)


        if step % 10_00 == 0:
            meanReturn, meanScore = evaluate(agent)
            print(f"Step {step} | V: {v_loss:.4f}, {mean_v:.4f} | Q: {q_loss:.4f}, {mean_q:.4f} | Pi: {pi_loss:.4f}, {mean_adv_wt:.4f} | Mean Score: {meanScore:.4f} | Mean Return: {meanReturn:.4f}")



train()