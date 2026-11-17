import gymnasium as gym
import minari
import torch
import numpy as np
import os
import json
import h5py

class DatasetLoader():
    GYM_ENV_MAP = {
        'halfcheetah': 'HalfCheetah-v4',
        'hopper': 'Hopper-v4',
        'walker2d': 'Walker2d-v4'
    }
    REF_SCORES = {
        'halfcheetah': (-280.18, 12135.0),
        'hopper': (-20.27, 3234.3),
        'walker2d': (1.63, 4592.3)
    }
    def __init__(self, datasetPath="mujoco/halfcheetah/medium-v0", device='cuda'):
        self.datasetPath = datasetPath
        lowerPath = datasetPath.lower()
        self.specialDataset = None
        if any(val in lowerPath for val in ['locomotion', 'halfcheetah', 'hopper', 'walker2d']):
            self.specialDataset = 'locomotion'
        elif 'antmaze' in lowerPath:
            self.specialDataset = 'antmaze'
        self.device = device if torch.cuda.is_available() else 'cpu'
        if datasetPath.endswith('.hdf5'):
            self.dataset = None
            self.env = gym.make(self.getEnv())
            self.evalEnv = gym.make(self.getEnv())
            self.data = self.flatten_legacy()
        else:
            self.dataset = minari.load_dataset(datasetPath, download=True)
            self.env = self.dataset.recover_environment()
            self.evalEnv = self.dataset.recover_environment(eval_env=True)
            self.data = self.flatten()


    def getEnv(self):
        for key,env in DatasetLoader.GYM_ENV_MAP.items():
            if key in f'{self.datasetPath}'.lower():
                return env

    def flatten(self):
        episodeReturns = []
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

        episodeReturnsArr = np.array(episodeReturns, dtype=np.float32)
        print(f'Mean return: {episodeReturnsArr.mean()}')
        print(f'min return: {episodeReturnsArr.min()}')
        print(f'max return: {episodeReturnsArr.max()}')
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

    def flatten_legacy(self):
        datasetpath = os.path.join("PaperDatasets", self.datasetPath)
        with h5py.File(datasetpath) as f:
            obs = f['observations'][:].astype(np.float32)
            actions = f['actions'][:].astype(np.float32)
            rewards = f['rewards'][:].astype(np.float32)
            terminations = f['terminals'][:].astype(bool).reshape(-1)
            timeouts = f['timeouts'][:].astype(bool).reshape(-1) if 'timeouts' in f else np.zeros_like(terminations)
            nextObs = f['next_observations'][:].astype(np.float32)
        episodeReturns = []
        currReturn = 0.0
        N = len(obs)

        for i in range(N):
            currReturn += float(rewards[i])
            isLastStep = (i==N-1) or terminations[i] or timeouts[i]
            if not isLastStep:
                if np.linalg.norm(obs[i+1]-nextObs[i]) > 1e-6:
                    isLastStep = True

            if isLastStep:
                episodeReturns.append(currReturn)
                currReturn = 0.0
        episodeReturnsArr = np.array(episodeReturns, dtype=np.float32)
        print(f'Mean return: {episodeReturnsArr.mean()}')
        print(f'min return: {episodeReturnsArr.min()}')
        print(f'max return: {episodeReturnsArr.max()}')
        terminations = terminations.astype(np.float32)

        rewards = self.preprocessRewards(rewards, episodeReturns)

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


    def getScores(self):
        for key,score in DatasetLoader.REF_SCORES.items():
            if key in f'{self.datasetPath}'.lower():
                return score

    def normalizedScore(self, raw_returns):
        try:
            assert(self.dataset is not None)
            return minari.get_normalized_score(self.dataset, raw_returns)*100.0
        except:
            # For halfcheetah, Todo: add dict for random and expert score
            r_random, r_expert = self.getScores()
            # r_expert = 12135.0
            # r_random = -280.18
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


def saveJson(data, fileName):
    with open(os.path.join('Outputs', fileName), 'w') as f:
        f.write(json.dumps(data, indent=2))

