import gym
import gym_pcgrl

import numpy as np
import math

import pdb

# render obs array as a string
render = lambda obs:print('\n'.join(["".join([str(i) for i in obs[j,:,0]]) for j in range(obs.shape[0])]))
# clean the input action
get_action = lambda a: a.item() if hasattr(a, "item") else a
# unwrap all the environments and get the PcgrlEnv
get_pcgrl_env = lambda env: env if "PcgrlEnv" in str(type(env)) else get_pcgrl_env(env.env)
# for the guassian attention
pdf = lambda x,mean,sigma: math.exp(-1/2 * math.pow((x-mean)/sigma,2))/math.exp(0)

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
Return a single array with all in it

can be stacked as Last Layer
"""
class ToFlat(gym.Wrapper):
    def __init__(self, game, names, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)
        length=0
        max_value=0
        for n in names:
            assert n in self.env.observation_space.spaces.keys(), 'This wrapper only works if your observation_space is spaces.Dict with the input names.'
            new_shape = self.env.observation_space[n]
            length += np.prod(new_shape)
            if self.env.observation_space[n].high.max() > max_value:
                max_value = self.env.observation_space[n].high.max()
        self.names = names
        self.observation_space = gym.spaces.Box(low=0, high=max_value, shape=(length,))

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
        concatenations = []
        for n in self.names:
            concatenations.append(obs[n].flatten())
        return np.concatentate(concatenations)

"""
Transform any object in the dictionary to one hot encoding

can be stacked
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

"""
Adding sum of the heatmap to the observation grid

can be stacked
"""
class AddChanges(gym.Wrapper):
    def __init__(self, game, is_map=True, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)

        assert 'heatmap' in self.env.observation_space.spaces.keys(), 'Heatmap does not exists in the environment space.'
        heatmap_obs = self.observation_space.spaces['heatmap']
        self.is_map = is_map

        if not self.is_map:
            self.observation_space.spaces['changes'] = gym.spaces.Box(low=[heatmap_obs.low.min()], high=[heatmap_obs.high.max()], dtype=heatmap_obs.dtype)
        else:
            self.observation_space.spaces['changes'] = gym.spaces.Box(low=heatmap_obs.low.min(), high=heatmap_obs.high.max(), shape=heatmap_obs.shape, dtype=heatmap_obs.dtype)

    def reset(self):
        obs = self.env.reset()
        return self.transform(obs)

    def step(self, action):
        action = get_action(action)
        obs, reward, done, info = self.env.step(action)
        obs = self.transform(obs)
        return obs, reward, done, info

    def transform(self, obs):
        obs['changes'] = obs['heatmap'].sum()
        if self.is_map:
            obs['changes'] = np.full(obs['heatmap'].shape, obs['changes'])
        return obs

"""
Returns reward at the end of the episode

Can be stacked
"""
class LateReward(gym.Wrapper):
    def __init__(self, game, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)
        self.acum_reward = 0

    def reset(self):
        self.acum_reward = 0
        return self.env.reset()

    def step(self, action):
        action = get_action(action)
        obs, reward, done, info = self.env.step(action)
        self.acum_reward += reward
        reward=[0,self.acum_reward][done]
        return obs, reward, done, info

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
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(h,w,dim))

    def reset(self):
        self.old_obs = self.env.reset()
        return self.old_obs

    def step(self, action):
        y, x, v = np.unravel_index(np.argmax(action), action.shape)
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
Normalize a certain attribute by the max and min values of its observation_space

can be stacked
"""
class Normalize(gym.Wrapper):
    def __init__(self, game, name, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)

        assert name in self.env.observation_space.spaces.keys(), 'This wrapper only works if you have a {} key'.format(name)
        self.name = name
        self.low = self.env.observation_space[self.name].low
        self.high = self.env.observation_space[self.name].high

        self.observation_space = gym.spaces.Dict({})
        for (k,s) in self.env.observation_space.spaces.items():
            self.observation_space.spaces[k] = s
        shape = self.observation_space[self.name].shape
        self.observation_space.spaces[self.name] = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)

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
        test = obs[self.name]
        obs[self.name] = (test - self.low).astype(np.float32) / (self.high - self.low).astype(np.float32)
        return obs

"""
Inverse the values a certain attribute of its observation_space

can be stacked
"""
class Inverse(gym.Wrapper):
    def __init__(self, game, name, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)

        assert name in self.env.observation_space.spaces.keys(), 'This wrapper only works if you have a {} key'.format(name)
        self.name = name
        self.low = self.env.observation_space[self.name].low
        self.high = self.env.observation_space[self.name].high

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
        test = obs[self.name]
        obs[self.name] = self.high - test + self.low
        return obs

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

"""
Add a 2D map with a window of 1s as pos key instead of normal pos
This wrapper only works on games with a position coordinate

can be stacked
"""
class PosImage(gym.Wrapper):
    def __init__(self, game, pos_size=1, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)

        assert 'pos' in self.env.observation_space.spaces.keys(), 'This wrapper only works for views that have a position'
        assert 'map' in self.env.observation_space.spaces.keys(), 'This wrapper only works for views that have a map'
        x, y = self.env.observation_space['map'].shape
        self.size = pos_size
        self.pad = pos_size//2
        self.observation_space = gym.spaces.Dict({})
        for (k,s) in self.env.observation_space.spaces.items():
            self.observation_space.spaces[k] = s
        self.observation_space.spaces['pos'] = gym.spaces.Box(low=0, high=1, shape=(x, y), dtype=np.float32)

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
        map = obs['map']
        x, y = obs['pos']

        pos = np.zeros_like(map)
        low_y,high_y=np.clip(y-self.pad,0,map.shape[0]),np.clip(y+(self.size-self.pad),0,map.shape[0])
        low_x,high_x=np.clip(x-self.pad,0,map.shape[1]),np.clip(x+(self.size-self.pad),0,map.shape[1])
        pos[low_y:high_y,low_x:high_x] = 1

        obs['pos'] = pos
        return obs

"""
Similar to the Image Wrapper but the values in the image
are sampled from gaussian distribution

Can be stacked
"""
class PosGaussianImage(PosImage):
    def __init__(self, game, pos_size=5, guassian_std=1, **kwargs):
        Image.__init__(self, game, pos_size, **kwargs)
        assert guassian_std > 0, 'gaussian distribution need positive standard deviation'
        self.guassian = guassian_std

    def transform(self, obs):
        shape = obs['map'].shape
        pos_x, pos_y = obs['pos']
        obs = Image.transform(self, obs)
        for y in range(min(self.pad + 1,shape[0]//2+1)):
            for x in range(min(self.pad + 1,shape[1]//2+1)):
                value = pdf(np.linalg.norm(np.array([x, y])), 0, self.guassian)
                obs_y, obs_x = pos_y+y,pos_x+x
                if obs_y >= 0 and obs_x >= 0 and obs_y < shape[1] and obs_x < shape[0]:
                    obs['pos'][obs_y][obs_x][1] *= value
                obs_y, obs_x = pos_y-y,pos_x+x
                if obs_y >= 0 and obs_x >= 0 and obs_y < shape[1] and obs_x < shape[0]:
                    obs['pos'][obs_y][obs_x][1] *= value
                obs_y, obs_x = pos_y+y,pos_x-x
                if obs_y >= 0 and obs_x >= 0 and obs_y < shape[1] and obs_x < shape[0]:
                    obs['pos'][obs_y][obs_x][1] *= value
                obs_y, obs_x = pos_y-y,pos_x-x
                if obs_y >= 0 and obs_x >= 0 and obs_y < shape[1] and obs_x < shape[0]:
                    obs['pos'][obs_y][obs_x][1] *= value
        return obs

################################################################################
#   Final used wrappers for the experiments
################################################################################

"""
The wrappers we use for our experiment
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
        # Adding changes to the channels
        env = AddChanges(env, True)
        # Cropping the changes to same size
        env = Cropped(env, crop_size, 0, 'changes')
        # Normalizing the changes to be between 0 and 1
        env = Normalize(env, 'changes')
        # Cropping the heatmap similar to the map
        env = Cropped(env, crop_size, 0, 'heatmap')
        # Normalize the heatmap
        env = Normalize(env, 'heatmap')
        # Final Wrapper has to be ToImage or ToFlat
        self.env = ToImage(env, ['map', 'heatmap', 'changes'])
        gym.Wrapper.__init__(self, self.env)

"""
This wrapper ignore location data, pretty useful with wide representation
"""
class ImagePCGRLWrapper(gym.Wrapper):
    def __init__(self, game, **kwargs):
        self.pcgrl_env = gym.make(game)
        self.pcgrl_env.adjust_param(**kwargs)
        # Normalize the heatmap
        env = Normalize(self.pcgrl_env, 'heatmap')
        # Adding changes to the channels
        env = AddChanges(env, True)
        # Normalize Changes
        env = Normalize(env, 'changes')
        # Transform to one hot encoding if not binary
        if 'binary' not in game:
            env = OneHotEncoding(env, 'map')
        # Final Wrapper has to be ToImage or ToFlat
        self.env = ToImage(env, ['map', 'heatmap', 'changes'])
        gym.Wrapper.__init__(self, self.env)

"""
Similar to the previous wrapper but the input now is 3D map (height, width, num_tiles)
"""
class ActionMapImagePCGRLWrapper(gym.Wrapper):
    def __init__(self, game, **kwargs):
        self.pcgrl_env = gym.make(game)
        self.pcgrl_env.adjust_param(**kwargs)
        # Add the action map wrapper
        env = ActionMap(self.pcgrl_env)
        # Normalize the heatmap
        env = Normalize(env, 'heatmap')
        # Adding changes to the channels
        env = AddChanges(env, True)
        # Normalize Changes
        env = Normalize(env, 'changes')
        # Transform to one hot encoding if not binary
        if 'binary' not in game:
            env = OneHotEncoding(env, 'map')
        # Final Wrapper has to be ToImage or ToFlat
        self.env = ToImage(env, ['map', 'heatmap', 'changes'])
        gym.Wrapper.__init__(self, self.env)

"""
Instead of cropping we are appending 1s in position layer
"""
class PositionImagePCGRLWrapper(gym.Wrapper):
    def __init__(self, game, pos_size, guassian_std=0, **kwargs):
        self.pcgrl_env = gym.make(game)
        self.pcgrl_env.adjust_param(**kwargs)
        # Normalize the heatmap
        env = Normalize(self.pcgrl_env, 'heatmap')
        # Adding changes to the channels
        env = AddChanges(env, True)
        # Normalize Changes
        env = Normalize(env, 'changes')
        # Transform to one hot encoding if not binary
        if 'binary' not in game:
            env = OneHotEncoding(env, 'map')
        # Transform the pos to image
        if guassian_std > 0:
            env = PosImage(env, pos_size, guassian_std)
        else:
            env = PosImage(env, pos_size)
        # Final Wrapper has to be ToImage or ToFlat
        self.env = ToImage(env, ['map', 'pos', 'heatmap', 'changes'])
        gym.Wrapper.__init__(self, self.env)

"""
Flat input for fully connected layers
"""
class FlatPCGRLWrapper(gym.Wrapper):
    def __init__(self, game, **kwargs):
        self.pcgrl_env = gym.make(game)
        self.pcgrl_env.adjust_param(**kwargs)
        # Normalize the heatmap
        env = Normalize(self.pcgrl_env, 'heatmap')
        # Adding changes to the channels
        env = AddChanges(env, False)
        # Normalize Changes
        env = Normalize(env, 'changes')
        # Normalize the position
        if 'pos' in self.pcgrl_env.observation_space.spaces.keys():
            env = Normalize(env, 'pos')
        # Transform to one hot encoding if not binary
        if 'binary' not in game:
            env = OneHotEncoding(env, 'map')
        # Final Wrapper has to be ToImage or ToFlat
        self.env = ToFlat(env, ['map', 'heatmap', 'pos', 'changes'])
        gym.Wrapper.__init__(self, self.env)
