#pip install tensorflow==1.15
#Install stable-baselines as described in the documentation

import gym
import gym_pcgrl
from gym_pcgrl import wrappers

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

def Cnn(image, **kwargs):
    activ = tf.nn.relu
    layer_1 = activ(conv(image, 'c1', n_filters=32, filter_size=3, stride=2, init_scale=np.sqrt(2), **kwargs)) # filter_size=3
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=3, stride=2, init_scale=np.sqrt(2), **kwargs)) #filter_size = 3
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))

class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs, cnn_extractor=Cnn, feature_extraction="cnn")

def main(game, representation, experiment_desc, env_func, steps, n_cpu):
    env_name = '{}-{}-v0'.format(game, representation)
    experiment = '{}_{}_{}'.format(game, representation, experiment_desc)

    if(n_cpu > 1):
        env = SubprocVecEnv([lambda: env_func(env_name) for i in range(n_cpu)])
    else:
        env = DummyVecEnv([lambda: env_func(env_name)])

    model = PPO2(CustomPolicy, env, verbose=1, tensorboard_log="./runs")
    model.learn(total_timesteps=int(steps), tb_log_name=experiment) #, callback=callback)
    model.save(experiment)

if __name__ == '__main__':
    game = 'binary'
    representation = 'narrow'
    experiment = 'limited_centered'
    n_cpu = 24
    steps = 1e8
    env = lambda game: wrappers.DefaultImagePCGRLWrapper(game, 28, random_tile=False)
    main(game, representation, experiment, env, steps, n_cpu)
