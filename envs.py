from gym_pcgrl import wrappers, conditional_wrappers
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from utils import RenderMonitor

def make_env(env_name, representation, rank=0, log_dir=None, **kwargs):
    '''
    Return a function that will initialize the environment when called.
    '''
    max_step = kwargs.get('max_step', None)
    render = kwargs.get('render', False)
    conditional = kwargs.get('conditional', False)
    def _thunk():
        if representation == 'wide':
            env = wrappers.ActionMapImagePCGRLWrapper(env_name, **kwargs)
        else:
            crop_size = kwargs.get('cropped_size', 28)
            env = wrappers.CroppedImagePCGRLWrapper(env_name, crop_size, **kwargs)
        if max_step is not None:
            env = wrappers.MaxStep(env, max_step)
        if log_dir is not None and kwargs.get('add_bootstrap', False):
            env = wrappers.EliteBootStrapping(env,
                                              os.path.join(log_dir, "bootstrap{}/".format(rank)))
        # RenderMonitor must come last
        if conditional:
            env = conditional_wrappers.ParamRew(env, cond_metrics=kwargs.pop('cond_metrics'), **kwargs)
            env.configure(**kwargs)
            env = conditional_wrappers.UniformNoiseyTargets(env, **kwargs)
        if render or log_dir is not None and len(log_dir) > 0:
            env = RenderMonitor(env, rank, log_dir, **kwargs)
        return env
    return _thunk

def make_vec_envs(env_name, representation, log_dir, **kwargs):
    '''
    Prepare a vectorized environment using a list of 'make_env' functions.
    '''
    n_cpu = kwargs.pop('n_cpu')
    if n_cpu > 1:
        env_lst = []
        for i in range(n_cpu):
            env_lst.append(make_env(env_name, representation, i, log_dir, **kwargs))
        env = SubprocVecEnv(env_lst)
    else:
        env = DummyVecEnv([make_env(env_name, representation, 0, log_dir, **kwargs)])
    return env


