import gymnasium as gym
import minari
import torch
import numpy as np

class DatasetLoader():
    def __init__(self, datasetPath="mujoco/halfcheetah/medium-v0", device='cuda'):
        lowerPath = datasetPath.lower()
        self.specialDataset = None
        if any(val in lowerPath for val in ['locomotion', 'halfcheetah', 'hopper', 'walker2d']):
            self.specialDataset = 'locomotion'
        elif 'antmaze' in lowerPath:
            self.specialDataset = 'antmaze'
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.dataset = minari.load_dataset(datasetPath, download=True)
        self.env = self.dataset.recover_environment()
        self.evalEnv = self.dataset.recover_environment(eval_env=True)
        self.data = self.flatten()

    def flatten(self):
        episodeReturns =[]
        obsList, nextObsList = [],[]
        actList, rewardsList, termList = [],[],[]
        for _episode in self.dataset.iterate_episodes():
            episode = self.processObs(_episode.observations)
            obsList.append(episode[:-1])
            nextObsList.append(episode[1:])
            actList.append(_episode.actions)
            rewardsList.append(_episode.rewards)
            termList.append(_episode.terminations)
            episodeReturns.append(np.sum(_episode.rewards))


        obs = np.concatenate(obsList, axis=0)
        nextObs = np.concatenate(nextObsList, axis=0)
        actions = np.concatenate(actList, axis=0)
        rewards = self.preprocessRewards(np.concatenate(rewardsList, axis=0), episodeReturns)
        terminations = np.concatenate(termList, axis=0)

        self.size = obs.shape[0]
        self.obsDim = obs.shape[1]
        self.actDim = actions.shape[1]
        self.data = {}
        for key,val in (('observations', obs), ('next_observations', nextObs), ('actions', actions), ('rewards', rewards), ('terminations', terminations)):
            self.data[key] = torch.tensor(val, dtype=torch.float32, device=self.device)
        return self.data

    def processObs(self, episode):
        if isinstance(episode, dict): # for antmaze
            return np.concatenate([episode['observation'], episode['desired_goal']], axis=1)
        else:
            return episode

    def preprocessRewards(self, rewards, episodeReturns):
        if self.specialDataset is None:
            return rewards
        elif self.specialDataset == 'locomotion':
            return rewards*(1000/(max(episodeReturns)-min(episodeReturns)))
        elif self.specialDataset == 'antmaze':
            return rewards-1.0
        return rewards

    def normalizedScore(self, raw_returns):
        try:
            return minari.get_normalized_score(self.dataset, raw_returns)*100.0
        except:
            # For halfcheetah, Todo: add dict for random ad expert score
            r_expert = 12135.0
            r_random = -280.18
            score = 100*((raw_returns-r_random)/(r_expert-r_random))
            return score

    def samplebatch(self, batchSize):
        idx = torch.randint(0, self.size, (batchSize,), device=self.device)
        sampleData = {}
        for key in ('observations', 'next_observations', 'actions', 'rewards', 'terminations'):
            sampleData[key] = self.data[key][idx]
        return sampleData

    def __str__(self):
        return f'''
        device: {self.device}
        size: {self.size},
        obsDim: {self.obsDim},
        actDim: {self.actDim},
        rewards mean: {self.data['rewards'].mean()},
        terminations sum: {self.data['terminations'].sum()}
        '''



halfCheetah = DatasetLoader()
print(halfCheetah)
