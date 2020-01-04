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
import re
import glob
import matplotlib.pyplot as plt


import pdb

n_steps = 0
log_dir = './'
best_mean_reward, n_steps = -np.inf, 0


def show_state(env, l, c, r, step=0, name="", info=""):
    fig = plt.figure(10)
    plt.clf()

    plt.title("{} | Step: {} Path: {} Changes: {} Regions: {}".format(name, step, l[-1], c[-1], r[-1]))

    ax1 = fig.add_subplot(1,4,1)
    ax1 = plt.imshow(env.render(mode='rgb_array'))
    plt.axis('off')

    ax2 = fig.add_subplot(1,4,2)
    ax2 = plt.plot(l)

    ax3 = fig.add_subplot(1,4,3)
    ax3 = plt.plot(c)

    ax4 = fig.add_subplot(1,4,4)
    ax4 = plt.plot(r)


    fig.set_figwidth(15)
    plt.tight_layout()

    plt.show()
   #display.clear_output(wait=True)
   #display.display(plt.gcf())


def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    # Print stats every 1000 calls
    if (n_steps + 1) % 10 == 0:
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
                _locals['self'].save(os.path.join(log_dir, 'best_model.pkl'))
        else:
            print('{} monitor entries'.format(len(x)))
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


def get_exp_name(game, representation, experiment, **kwargs):
    exp_name = '{}_{}'.format(game, representation)
    change_percentage = kwargs.get('change_percentage', None)
    path_length = kwargs.get('target_path', None)
    if change_percentage is not None:
        exp_name = '{}_chng{}_pth{}'.format(exp_name, change_percentage, path_length)
    if experiment is not None:
        exp_name = '{}_{}'.format(exp_name, experiment)
    return exp_name


def max_exp_idx(exp_name):
    log_dir = os.path.join("./runs", exp_name)
    log_files = glob.glob('{}*'.format(log_dir))
    if len(log_files) == 0:
        n = 1
    else:
        log_ns = [re.search('_(\d+)', f).group(1) for f in log_files]
        n = max(log_ns)
    return n


def main(game, representation, experiment, steps, n_cpu, render, logging, **kwargs):
    env_name = '{}-{}-v0'.format(game, representation)
    exp_name = get_exp_name()
    if representation == 'wide':
        policy = FullyConvPolicy
    else:
        policy = CustomPolicy

    global log_dir
    n = max_exp_idx(exp_name)
    n = n + 1
    log_dir = '{}_{}'.format(log_dir, n)
    log_dir = '{}_{}'.format(log_dir, 'log')
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
        model.learn(total_timesteps=int(steps), tb_log_name=exp_name)
    else:
        model.learn(total_timesteps=int(steps), tb_log_name=exp_name, callback=callback)
    model.save(experiment)


def infer(game, representation, experiment, **kwargs):
    kwargs = {
            **kwargs,
            'inference': True,
            'render': True,
            }
    env_name = '{}-{}-v0'.format(game, representation)
    exp_name = get_exp_name(game, representation, experiment, **kwargs)
    global log1r
    n = max_exp_idx(exp_name)
    log_dir = '{}_{}_log'.format(exp_name, n)
    log_dir = os.path.join('runs', log_dir, 'best_model.pkl')
    model = PPO2.load(log_dir)
    log_dir = None
   #log_dir = os.path.join(log_dir, 'eval')
    kwargs = {
            **kwargs,
            'change_percentage': 1,
            'target_path': 98,
            }
    env = DummyVecEnv([make_env(env_name, representation, 0, log_dir, **kwargs)])
    obs = env.reset()
    n_step = 0
    path_length = []
    changes = []
    regions = []
    while True:
        if n_step >= 1200:
            obs = env.reset()
            n_step = 0
        else:
            action = get_action(obs, env, model)
            obs, rewards, dones, info = env.step(action)
            path_length.append(info[0]['path-length'])
            changes.append(info[0]['changes'])
            regions.append(info[0]['regions'])
            print(info)
            if dones:
                show_state(env, path_length, changes, regions, n_step)
            n_step += 1


def get_action(obs, env, model, action_type=True):
    action = None
    if action_type == 0:
        action, _ = model.predict(obs)
    elif action_type == 1:
        action_prob = model.action_probability(obs)[0]
        action = np.random.choice(a=list(range(len(action_prob))), size=1, p=action_prob)
    else:
        action = np.array([env.action_space.sample()])
    return action


"""
Wrapper for the environment to save data in .csv files.
"""
class RenderMonitor(Monitor):
    def __init__(self, env, rank, log_dir, **kwargs):
        self.log_dir = log_dir
        self.rank = rank
        self.render_gui = kwargs.get('render', False)
        self.render_rank = kwargs.get('render_rank', 0)
        if log_dir is not None:
            log_dir = os.path.join(log_dir, str(rank))
        Monitor.__init__(self, env, log_dir)

    def step(self, action):
        if self.render_gui and self.rank == self.render_rank:
            self.render()
        return Monitor.step(self, action)

def make_env(env_name, representation, rank, log_dir, **kwargs):
    def _thunk():
        if representation == 'wide':
            env = wrappers.ActionMapImagePCGRLWrapper(env_name, **kwargs)
        else:
            env = wrappers.CroppedImagePCGRLWrapper(env_name, 28, **kwargs)
        if log_dir != None and len(log_dir) > 0 or kwargs.get('inference', False):
            env = RenderMonitor(env, rank, log_dir, **kwargs)
        return env
    return _thunk


game = 'binary'
representation = 'wide'
experiment = None
n_cpu = 96
steps = 5e7
render = True
logging = True
kwargs = {
        # specific to binary:
        'change_percentage': 0.2,
        'target_path': 48,
        }


def enjoy():
    infer(game, representation, experiment, **kwargs)


if __name__ == '__main__':
    main(game, representation, experiment, steps, n_cpu, render, logging, **kwargs)
