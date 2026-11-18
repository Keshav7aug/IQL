from utils import DatasetLoader, saveJson
from agent import IQLAgent
import torch
import numpy as np
import time
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

def train(steps=1_000_000, datasetPath='halfcheetah_medium-v2.hdf5', baseline=False):
    startTime = time.time()
    loader = DatasetLoader(datasetPath=datasetPath,device='cuda')
    tau = 0.7 # locomotion
    if baseline:
        tau = 0.5
    elif loader.specialDataset != 'antmaze':
        tau = 0.9
    agent = IQLAgent(
        loader=loader,
        lr=3e-4,
        gamma=0.99,
        tau=tau,
        beta=3.0 if loader.specialDataset == 'locomotion' else 10.0
    )

    trainingData = []
    for step in range(1,steps+1):
        batch = loader.samplebatch(batchSize=256)


        v_loss, mean_v = agent.update_v(batch=batch)
        pi_loss, mean_adv_wt = agent.update_policy(batch=batch)
        q_loss, mean_q = agent.update_q(batch=batch)


        if step % 1_000 == 0:
            meanReturn, meanScore = evaluate(agent)
            print(f"Step {step} | V: {v_loss:.4f}, {mean_v:.4f} | Q: {q_loss:.4f}, {mean_q:.4f} | Pi: {pi_loss:.4f}, {mean_adv_wt:.4f} | Mean Score: {meanScore:.4f} | Mean Return: {meanReturn:.4f}")
            trainingData.append({
                'Step': step,
                'V Loss': round(v_loss, 4),
                'V Mean': round(mean_v, 4),
                'Q Loss': round(q_loss, 4),
                'Q Mean': round(mean_q, 4),
                'Pi Loss': round(pi_loss, 4),
                'Pi Mean': round(mean_adv_wt, 4),
                'Mean Score': round(meanScore, 4),
                'Mean Return': round(meanReturn, 4),
            })
    endTime = time.time()    
    metadata = {
        "TotalTime": endTime-startTime
        }
    saveJson([metadata]+trainingData, f"{loader.datasetPath.replace('/', '_')}_{steps}_Tau{agent.tau}.json")
datasets = ('halfcheetah_medium_replay-v2.hdf5', 'halfcheetah_medium-v2.hdf5', 'hopper_medium-v2.hdf5', 'walker2d_medium-v2.hdf5', 'mujoco/halfcheetah/medium-v0', 'mujoco/hopper/medium-v0', 'mujoco/walker2d/medium-v0', 'D4RL/antmaze/medium-play-v1', 'D4RL/antmaze/medium-diverse-v1')
baseLines = (True, False)
for datasetPathTest in datasets:
    for isbaseline in baseLines:
        print(datasetPathTest, isbaseline)
        train(steps=1_000_000, datasetPath=datasetPathTest, baseline=isbaseline)
# train()