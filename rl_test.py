#pip install tensorflow==1.15
#Install stable-baselines as described in the documentation

import gym
import gym_pcgrl

from stable_baselines.common.policies import MlpPolicy, CnnPolicy, FeedForwardPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines.a2c.utils import conv, linear, conv_to_fc
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.bench import Monitor
from stable_baselines import PPO2

import tensorflow as tf
import numpy as np

import pdb

log_dir = './'
best_mean_reward, n_steps = -np.inf, 0

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    # Print stats every 1000 calls
    if (n_steps + 1) % 1 == 0:
        # Evaluate policy training performance
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        pdb.set_trace()
        if len(x) > 0:
            pdb.set_trace()
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(log_dir + 'best_model.pkl')
    n_steps += 1
    # Returning False will stop training early
    return True

class PCGRL(gym.Wrapper):
    def __init__(self, game):
        self.env = gym.make(game)
        self.env.adjust_param(random_tile=False, max_iterations = 1000)
        gym.Wrapper.__init__(self, self.env)

        #self.observation_space = gym.spaces.Box(low=0, high=1, shape=(198,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(14, 14, 1), dtype=np.uint8)

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
        #flatten
        #map = obs['map'].flatten()
        #pos = obs['pos']/168
        #return np.concatenate([map, pos])

        #image
        map = obs['map']
        x, y = obs['pos']
        pos = np.zeros([14,14,1])
        #pos = np.zeros_like(map)
        pos[y][x] = 1
        #return np.stack([map, pos], 2)
        return pos

def Cnn(image, **kwargs):
    activ = tf.nn.relu
    layer_1 = activ(conv(image, 'c1', n_filters=32, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))

class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs, cnn_extractor=Cnn, feature_extraction="cnn")

def main(game, n_cpu):
    # multiprocess environment
    #env = SubprocVecEnv([(lambda worker: lambda: Monitor(PCGRL(game), log_dir, allow_early_resets=True))(i) for i in range(n_cpu)])
    env = SubprocVecEnv([lambda: PCGRL(game) for i in range(n_cpu)])
    #env = DummyVecEnv([lambda: PCGRL()])

    model = PPO2(CustomPolicy, env, verbose=0, tensorboard_log="./runs")
    model.learn(total_timesteps=int(5e6), tb_log_name="Logging_Test") #, callback=callback)
    #model.save("ppo_binary")

    #del model # remove to demonstrate saving and loading

    #model = PPO2.load("ppo_binary")

# Enjoy trained agent
#obs = env.reset()
#while True:
#    action, _states = model.predict(obs)
#    obs, rewards, dones, info = env.step(action)
#    env.render()

if __name__ == '__main__':
    main('binary-narrow-v0', 24)
