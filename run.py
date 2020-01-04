#pip install tensorflow==1.15
#Install stable-baselines as described in the documentation

import gym
import gym_pcgrl
from gym_pcgrl import wrappers

from stable_baselines.common.policies import MlpPolicy, CnnPolicy, ActorCriticPolicy, FeedForwardPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines.a2c.utils import conv, linear, conv_to_fc
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.bench import Monitor
from stable_baselines import PPO2

import tensorflow as tf
import numpy as np
import os
import shutil

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
    if (n_steps + 1) % 1000 == 0:
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 100:
           #pdb.set_trace()
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, we save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(os.path.join(log_dir + 'best_model.pkl'))
        else:
           #print('{} monitor entries'.format(len(x)))
            pass
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

def FullyConv(image, **kwargs):
    activ = tf.nn.relu
    x = activ(conv(image, 'c1', n_filters=32, filter_size=3, stride=1,
        pad='SAME', init_scale=np.sqrt(2)))
    x = activ(conv(x, 'c2', n_filters=64, filter_size=3, stride=1,
        pad='SAME', init_scale=np.sqrt(2)))
    x = activ(conv(x, 'c3', n_filters=64, filter_size=3, stride=1,
        pad='SAME', init_scale=np.sqrt(2)))
   #return x
    act = conv_to_fc(x)
    val = activ(conv(x, 'v1', n_filters=64, filter_size=3, stride=2,
        init_scale=np.sqrt(2)))
    val = activ(conv(val, 'v2', n_filters=64, filter_size=3, stride=2,
        init_scale=np.sqrt(3)))
    val = activ(conv(val, 'v3', n_filters=64, filter_size=3, stride=2,
        init_scale=np.sqrt(2), pad='SAME'))
   #val = activ(conv(val, 'v4', n_filters=64, filter_size=1, stride=1,
   #    init_scale=np.sqrt(2)))
    val = conv_to_fc(x)
    return act, val


class FullyConvPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(FullyConvPolicy, self).__init__(*args, **kwargs)
        with tf.variable_scope("model", reuse=kwargs['reuse']):
            pi_latent, vf_latent = FullyConv(self.processed_obs, **kwargs)
            self._value_fn = linear(vf_latent, 'vf', 1)
            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})

class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs, cnn_extractor=Cnn, feature_extraction="cnn")

def main(game, representation, experiment, steps, n_cpu, render, logging, **kwargs):
    env_name = '{}-{}-v0'.format(game, representation)
    exp_name = '{}_{}_{}'.format(game, representation, experiment)
    change_percentage = kwargs.get('change_percentage', None)
    path_length = kwargs.get('path_length', None)
    if change_percentage is not None:
        exp_name = exp_name + '_chng{}_pth{}'.format(change_percentage, path_length)

    if representation == 'wide':
        policy = FullyConvPolicy
    else:
        policy = CustomPolicy

    global log_dir
    log_dir = os.path.join("./runs", exp_name)
    # write monitors to folder based on 'experiment'
    # (would be better off in same folder as tf data)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)
    kwargs = {
        **kwargs,
        'render_rank': 0,
        'render': render,
    }
    if not logging:
        log_dir = None
    if(n_cpu > 1):
        env_lst = []
        for i in range(n_cpu):
            env_lst.append(make_env(env_name, representation, i, log_dir, **kwargs))
        env = SubprocVecEnv(env_lst)
    else:
        env = DummyVecEnv([make_env(env_name, representation, 0, log_dir, **kwargs)])
    model = PPO2(policy, env, verbose=1, tensorboard_log="./runs")
    if not logging:
        model.learn(total_timesteps=int(steps), tb_log_name=experiment)
    else:
        model.learn(total_timesteps=int(steps), tb_log_name=experiment, callback=callback)
    model.save(experiment)


"""
Wrapper for the environment to save data in .csv files.
"""
class RenderMointer(Monitor):
    def __init__(self, env, rank, log_dir, **kwargs):
        self.log_dir = log_dir
        self.rank = rank
        self.render_gui = kwargs.get('render', False)
        self.render_rank = kwargs.get('render_rank', 0)
        log_dir = os.path.join(log_dir, str(rank))
        Monitor.__init__(self, env, log_dir)

    def step(self, action):
        if self.render_gui and self.rank == self.render_rank:
            self.render()

        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

def make_env(env_name, representation, rank, log_dir, **kwargs):
    def _thunk():
        if representation == 'wide':
            env = wrappers.ActionMapImagePCGRLWrapper(env_name, **kwargs)
        else:
            env = wrappers.CroppedImagePCGRLWrapper(env_name, 28, **kwargs)
        if log_dir != None and len(log_dir) > 0:
            env = RenderMointer(env, rank, log_dir, **kwargs)
        return env
    return _thunk

if __name__ == '__main__':
    game = 'binary'
    representation = 'wide'
    experiment = '0'
    n_cpu = 48
    steps = 5e7
    render = True
    logging = True
    kwargs = {
            # specific to binary:
            'change_percentage': 0.2,
            'path_length': 48,
            }
    main(game, representation, experiment, steps, n_cpu, render, logging, **kwargs)
