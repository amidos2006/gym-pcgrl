import gym
import gym_pcgrl

import numpy as np

import pdb

#render obs array as a string
render = lambda obs:print('\n'.join(["".join([str(i) for i in obs[j,:,0]]) for j in range(obs.shape[0])]))

'''
Crops and centers the view around the agent
The crop size can be larger than the actual view, it just pads the outside
This wrapper only works on games with a position coordinate
'''
class Cropped(gym.Wrapper):
    def __init__(self, game, crop_size, **kwargs):
        self.env = gym.make(game)
        self.env.adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)

        assert 'pos' in self.env.observation_space.spaces.keys(), 'This wrapper only works for views thave have a position'
        self.size = crop_size
        self.pad = crop_size//2
        self.pad_value = self.env.get_border_tile()
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(crop_size, crop_size, 1), dtype=np.uint8)

    def step(self, action):
        action = action.item()
        obs, reward, done, info = self.env.step(action)
        obs = self.transform(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self.transform(obs)
        return obs

    def transform(self, obs):
        map = obs['map']
        x, y = obs['pos']

        padded = np.pad(map, self.pad, constant_values=self.pad_value)
        cropped = padded[y:y+self.size, x:x+self.size]
        return np.expand_dims(cropped, 2)


'''
Provides an image of the map with a layer for position
This wrapper only works on games with a position coordinate
'''
class Image(gym.Wrapper):
    def __init__(self, game, **kwargs):
        self.env = gym.make(game)
        self.env.adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)

        assert 'pos' in self.env.observation_space.spaces.keys(), 'This wrapper only works for views that have a position'
        x, y = self.env.observation_space['map'].shape
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(x, y, 2), dtype=np.unit8)

    def step(self, action):
        action = action.item()
        obs, reward, done, info = self.env.step(action)
        obs = self.transform(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self.transform(obs)
        return obs

    def transform(self, obs):
        map = obs['map']
        x, y = obs['pos']

        pos = np.zero_like(map)
        pos[y][x] = 1
        return np.stack([map, pos], 2)

'''
Displays the view as a single vector
'''
class Flat(gym.Wrapper):
    def __init__(self, game, **kwargs):
        self.env = gym.make(game)
        self.env.adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)

        self.x, self.y = self.env.observation_space['map'].shape
        if('pos' in self.env.observation_space.spaces.keys()):
            self.pos = True
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(2*self.x*self.y,), dtype=np.unit8)
        else:
            self.pos = False
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.x*self.y,), dtype=np.unit8)

    def step(self, action):
        action = action.item()
        obs, reward, done, info = self.env.step(action)
        obs = self.transform(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self.transform(obs)
        return obs

    def transform(self, obs):
        map = obs['map'].flatten()
        if(self.pos):
            pos = np.zero_like(map)
            x, y = obs['pos']
            pos[y*self.x + x] = 1
            map = np.concatentate([map, pos])
        return map

