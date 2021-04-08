import math
import os
from opensimplex import OpenSimplex
from pdb import set_trace as T

import gym
import numpy as np
from gym_pcgrl.envs.helper import get_int_prob, get_string_map

# clean the input action
get_action = lambda a: a.item() if hasattr(a, "item") else a
# unwrap all the environments and get the PcgrlEnv
get_pcgrl_env = lambda env: env if "PcgrlEnv" in str(type(env)) else get_pcgrl_env(env.env)


class MaxStep(gym.Wrapper):
    """
    Wrapper that resets environment only after a certain number of steps.
    """
    def __init__(self, game, max_step):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
       #get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)

        self.max_step = max_step
        self.unwrapped.max_step = max_step
        self.n_step = 0
        gym.Wrapper.__init__(self, self.env)

    def step(self, action):
         obs, reward, done, info = self.env.step(action)
         self.n_step += 1

         if self.n_step == self.max_step:
             done = True
         else:
             done = False

         return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.n_step = 0

        return obs

"""
Return a Box instead of dictionary by stacking different similar objects

Can be stacked as Last Layer
"""
class ToImage(gym.Wrapper):
    def __init__(self, game, names, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)
        self.shape = None
        depth=0
        max_value = 0

        for n in names:
            assert n in self.env.observation_space.spaces.keys(), 'This wrapper only works if your observation_space is spaces.Dict with the input names.'

            if self.shape == None:
                self.shape = self.env.observation_space[n].shape
            new_shape = self.env.observation_space[n].shape
            depth += 1 if len(new_shape) <= 2 else new_shape[2]
            assert self.shape[0] == new_shape[0] and self.shape[1] == new_shape[1], 'This wrapper only works when all objects have same width and height'

            if self.env.observation_space[n].high.max() > max_value:
                max_value = self.env.observation_space[n].high.max()
        self.names = names

        self.observation_space = gym.spaces.Box(low=0, high=max_value,shape=(self.shape[0], self.shape[1], depth))

    def step(self, action):
        action = get_action(action)
        obs, reward, done, info = self.env.step(action)
        obs = self.transform(obs)

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self.transform(obs)

        return obs

    def transform(self, obs):
        final = np.empty([])

        for n in self.names:
            if len(final.shape) == 0:
                final = obs[n].reshape(self.shape[0], self.shape[1], -1)
            else:
                final = np.append(final, obs[n].reshape(self.shape[0], self.shape[1], -1), axis=2)

        return final

"""
Transform any object in the dictionary to one hot encoding

can be stacked
       - name: name of the action subspace to one-hot encode. E.g., "map" 
"""
class OneHotEncoding(gym.Wrapper):
    def __init__(self, game, name, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)

        assert name in self.env.observation_space.spaces.keys(), 'This wrapper only works for representations thave have a {} key'.format(name)
        self.name = name

        self.observation_space = gym.spaces.Dict({})

        for (k,s) in self.env.observation_space.spaces.items():
            self.observation_space.spaces[k] = s
        new_shape = []
        shape = self.env.observation_space[self.name].shape
        self.dim = self.observation_space[self.name].high.max() - self.observation_space[self.name].low.min() + 1

        for v in shape:
            new_shape.append(v)
        new_shape.append(self.dim)
        self.observation_space.spaces[self.name] = gym.spaces.Box(low=0, high=1, shape=new_shape, dtype=np.uint8)

    def step(self, action):
        action = get_action(action)
        obs, reward, done, info = self.env.step(action)
        obs = self.transform(obs)

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self.transform(obs)

        return obs

    def transform(self, obs):
        old = obs[self.name]
        obs[self.name] = np.eye(self.dim)[old]

        return obs
    
    def get_one_hot_map(self):
        obs = {'map': self.env._rep._map}
        return self.transform(obs)

"""
Transform the input space to a 3D map of values where the argmax value will be applied

can be stacked
"""
class ActionMap(gym.Wrapper):
    def __init__(self, game, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)

        assert 'map' in self.env.observation_space.spaces.keys(), 'This wrapper only works if you have a map key'
        self.old_obs = None
        self.one_hot = len(self.env.observation_space['map'].shape) > 2
        w, h, dim = 0, 0, 0

        if self.one_hot:
            h, w, dim = self.env.observation_space['map'].shape
        else:
            h, w = self.env.observation_space['map'].shape
            dim = self.env.observation_space['map'].high.max()
        self.h = self.unwrapped.h = h
        self.w = self.unwrapped.w = w
        self.dim = self.unwrapped.dim = self.env.get_num_tiles()
       #self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(h,w,dim))
        self.action_space = gym.spaces.Discrete(h*w*self.dim)

    def reset(self):
        self.old_obs = self.env.reset()

        return self.old_obs

    def step(self, action):
       #y, x, v = np.unravel_index(np.argmax(action), action.shape)
#       y, x, v = np.unravel_index(action, (self.h, self.w, self.dim))
        v, x, y = np.unravel_index(action, (self.dim, self.w, self.h))

        if 'pos' in self.old_obs:
            o_x, o_y = self.old_obs['pos']

            if o_x == x and o_y == y:
                obs, reward, done, info = self.env.step(v)
            else:
                o_v = self.old_obs['map'][o_y][o_x]

                if self.one_hot:
                    o_v = o_v.argmax()
                obs, reward, done, info = self.env.step(o_v)
        else:
            obs, reward, done, info = self.env.step([x, y, v])
        self.old_obs = obs

        return obs, reward, done, info

"""
Crops and centers the view around the agent and replace the map with cropped version
The crop size can be larger than the actual view, it just pads the outside
This wrapper only works on games with a position coordinate

can be stacked
"""
class Cropped(gym.Wrapper):
    def __init__(self, game, crop_size, pad_value, name, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)

        assert 'pos' in self.env.observation_space.spaces.keys(), 'This wrapper only works for representations thave have a position'
        assert name in self.env.observation_space.spaces.keys(), 'This wrapper only works if you have a {} key'.format(name)
        assert len(self.env.observation_space.spaces[name].shape) == 2, "This wrapper only works on 2D arrays."
        self.name = name
        self.size = crop_size
        self.pad = crop_size//2
        self.pad_value = pad_value

        self.observation_space = gym.spaces.Dict({})

        for (k,s) in self.env.observation_space.spaces.items():
            self.observation_space.spaces[k] = s
        high_value = self.observation_space[self.name].high.max()
        self.observation_space.spaces[self.name] = gym.spaces.Box(low=0, high=high_value, shape=(crop_size, crop_size), dtype=np.uint8)

    def step(self, action):
        action = get_action(action)
        obs, reward, done, info = self.env.step(action)
        obs = self.transform(obs)

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self.transform(obs)

        return obs

    def transform(self, obs):
        map = obs[self.name]
        x, y = obs['pos']

        #View Centering
        padded = np.pad(map, self.pad, constant_values=self.pad_value)
        cropped = padded[y:y+self.size, x:x+self.size]
        obs[self.name] = cropped

        return obs

################################################################################
#   Final used wrappers for the experiments
################################################################################

"""
The wrappers we use for narrow and turtle experiments
"""
class CroppedImagePCGRLWrapper(gym.Wrapper):
    def __init__(self, game, crop_size, **kwargs):
        self.pcgrl_env = gym.make(game)
        self.pcgrl_env.adjust_param(**kwargs)
        # Cropping the map to the correct crop_size
        env = Cropped(self.pcgrl_env, crop_size, self.pcgrl_env.get_border_tile(), 'map')
        # Transform to one hot encoding if not binary

        if 'binary' not in game:
            env = OneHotEncoding(env, 'map')
        # Indices for flatting
        flat_indices = ['map']
        # Final Wrapper has to be ToImage or ToFlat
        self.env = ToImage(env, flat_indices)
        gym.Wrapper.__init__(self, self.env)

"""
Similar to the previous wrapper but the input now is the index in a 3D map (height, width, num_tiles) of the highest value
Used for wide experiments
"""
class ActionMapImagePCGRLWrapper(gym.Wrapper):
    def __init__(self, game, **kwargs):
        self.pcgrl_env = gym.make(game)
        self.pcgrl_env.adjust_param(**kwargs)
        # Indices for flatting
        flat_indices = ['map']
        env = self.pcgrl_env
        # Add the action map wrapper
        env = ActionMap(env)
        # Transform to one hot encoding if not binary

        if 'binary' not in game:
            env = OneHotEncoding(env, 'map')
        # Final Wrapper has to be ToImage or ToFlat
        self.env = ToImage(env, flat_indices)
        gym.Wrapper.__init__(self, self.env)


# This precedes the ParamRew wrapper so we only worry about the map as observation
class CAactionWrapper(gym.Wrapper):
    def __init__(self, game, **kwargs):
        self.pcgrl_env = gym.make(game)
        self.pcgrl_env.adjust_param(**kwargs)
        super().__init__(self.pcgrl_env)
        # Indices for flatting
        flat_indices = ['map']
        env = self.pcgrl_env
        self.n_tile_types = env.action_space.nvec[2]
        width = env._prob._width
        height = env._prob._height
        self.n_ca_tick = 0
        # Add the action map wrapper
#       env = ActionMap(env)
        # Transform to one hot encoding if not binary

        # we need the observation to be one-hot, so we can reliably separate map from control observations for NCA skip connection
        env = OneHotEncoding(env, 'map')
        # Final Wrapper has to be ToImage or ToFlat
        self.env = ToImage(env, flat_indices)
        gym.Wrapper.__init__(self, self.env)
        # NOTE: check this insanity out so cool
        self.action_space = self.pcgrl_env.action_space = gym.spaces.MultiDiscrete([self.n_tile_types] * width * height)
#       self.action_space = self.pcgrl_env.action_space = gym.spaces.Box(low=0, high=1, shape=(self.n_tile_types, width, height))
        self.last_action = None
        self.INFER = kwargs.get('infer')

    def step(self, action):
        env = self.env
        pcgrl_env = self.pcgrl_env
        action = action.reshape(pcgrl_env._rep._map.shape).astype(int)
#       obs = get_one_hot_map(action, self.n_tile_types)
        obs = action.reshape(1, *action.shape)
        obs = obs.transpose(1, 2, 0)
        pcgrl_env._rep._map = obs.squeeze(-1)
        obs = self.env.get_one_hot_map()
        print(obs['map'][:,:,-1:].transpose(1, 2, 0))
        self.n_ca_tick += 1
        if self.n_ca_tick <= 50:  # or (action == self.last_action).all():
            done = False
        else:
            done = True
        self.env._rep_stats = pcgrl_env._prob.get_stats(get_string_map(action, pcgrl_env._prob.get_tile_types()))
        pcgrl_env.metrics = env.metrics = self.env._rep_stats
        self.last_action = action

        return obs['map'], 0, done, {}

    def reset(self):
        self.last_action = None
        self.n_ca_tick = 0
#       if self.pcgrl_env.unwrapped._rep._map is None:
#           # get map dimensions (sloppy)
#           super().reset()
#       # load up our initial state (empty)
#       self.env.unwrapped._rep._random_start = True
#       init_state = np.zeros(self.unwrapped._rep._map.shape).astype(np.uint8)
#       self.unwrapped._rep._old_map = init_state
        obs = self.env.reset()
#       self.pcgrl_env._map = self.env._rep._map

        self.render()
#       obs = self.env.get_one_hot_map()
        
        return obs
