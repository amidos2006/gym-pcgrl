"""
Helper functions for train, infer, and eval modules.
"""
from pdb import set_trace as T
import os
import re
import glob
import numpy as np
from stable_baselines3.ppo import PPO
from stable_baselines3.common.monitor import Monitor
from gym_pcgrl import wrappers, conditional_wrappers
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

class RenderMonitor(Monitor):
    """
    Wrapper for the environment to save data in .csv files.
    """
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

def get_exp_name(game, representation, experiment, **kwargs):
    exp_name = '{}_{}'.format(game, representation)
    if experiment is not None:
        exp_name = '{}_{}'.format(exp_name, experiment)
    return exp_name

def max_exp_idx(exp_name):
    log_dir = os.path.join("./runs", exp_name)
    log_files = glob.glob('{}*'.format(log_dir))
    if len(log_files) == 0:
        n = 0
    else:
        log_ns = [re.search('_(\d+)', f).group(1) for f in log_files]
        n = max(log_ns)
    return int(n)

def load_model(log_dir):
    model_path = os.path.join(log_dir, 'latest_model.pkl')
    if not os.path.exists(model_path):
        model_path = os.path.join(log_dir, 'latest_model.zip')
    if not os.path.exists(model_path):
        model_path = os.path.join(log_dir, 'best_model.pkl')
    if not os.path.exists(model_path):
        model_path = os.path.join(log_dir, 'best_model.zip')
    if not os.path.exists(model_path):
        files = [f for f in os.listdir(log_dir) if '.pkl' in f or '.zip' in f]
        if len(files) > 0:
            # selects the last file listed by os.listdir
            model_path = os.path.join(log_dir, np.random.choice(files))
        else:
            raise Exception('No models are saved')
    model = PPO.load(model_path)
    return model
