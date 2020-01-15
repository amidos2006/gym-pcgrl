from gym_pcgrl import wrappers

from stable_baselines import PPO2
from stable_baselines.bench import Monitor

import os
import re
import glob

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

def make_env(env_name, representation, rank=0, log_dir=None, **kwargs):
    def _thunk():
        if representation == 'wide':
            env = wrappers.ActionMapImagePCGRLWrapper(env_name, **kwargs)
        else:
            crop_size = kwargs.get('cropped_size', 28)
            env = wrappers.CroppedImagePCGRLWrapper(env_name, crop_size, **kwargs)
        if log_dir != None and kwargs.get('add_bootstrap', False):
            env = wrappers.BootStrapping(env, os.path.join(log_dir,"bootstrap{}/".format(rank)))
        if log_dir != None and len(log_dir) > 0:
            env = RenderMonitor(env, rank, log_dir, **kwargs)
        return env
    return _thunk

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
            model_path = os.path.join(log_dir, f)
        else:
            raise 'No models are saved'
    model = PPO2.load(model_path)
    return model
