"""
Run a trained agent for qualitative analysis.
"""
from utils import get_exp_name, max_exp_idx, load_model, make_vec_envs, get_action

def infer(game, representation, experiment, infer_kwargs, **kwargs):
    """
     - max_trials: The number of trials per evaluation.
     - infer_kwargs: Args to pass to the environment.
    """
    infer_kwargs = {
            **infer_kwargs,
            'inference': True,
            'render': True,
            }
    max_trials = kwargs.get('max_trials', -1)
    n = kwargs.get('n', None)
    env_name = '{}-{}-v0'.format(game, representation)
    exp_name = get_exp_name(game, representation, experiment, **kwargs)
    if n is None:
        n = max_exp_idx(exp_name)
    if n == 0:
        raise Exception('Did not find ranked saved model of experiment: {}'.format(exp_name))
    log_dir = 'runs/{}_{}_{}'.format(exp_name, n, 'log')
    model = load_model(log_dir)
    # no log dir, 1 parallel environment
    env = make_vec_envs(env_name, representation, None, 1, **infer_kwargs)
    obs = env.reset()
    # Record final values of each trial
    if 'binary' in env_name:
        path_length = []
        changes = []
        regions = []
        infer_info = {
            'path_length': [],
            'changes': [],
            'regions': [],
            }
    n_trials = 0
    while n_trials != max_trials:
       #action = get_action(obs, env, model)
        action, _ = model.predict(obs)
        obs, _, dones, info = env.step(action)
        if 'binary' in env_name:
            path_length.append(info[0]['path-length'])
            changes.append(info[0]['changes'])
            regions.append(info[0]['regions'])
        print(info)
       #for p, v in model.get_parameters().items():
       #    print(p, v.shape)
        if dones:
           #show_state(env, path_length, changes, regions, n_step)
            if 'binary' in env_name:
                infer_info['path_length'] = path_length[-1]
                infer_info['changes'] = changes[-1]
                infer_info['regions'] = regions[-1]
            n_trials += 1
    return infer_info

# For locating trained model
game = 'binary'
representation = 'wide'
experiment = 'LongConv'
kwargs = {
       #'change_percentage': 1,
       #'target_path': 105,
       #'n': 4, # rank of saved experiment (by default, n is max possible)
        }

# For inference
infer_kwargs = {
       #'change_percentage': 1,
       #'target_path': 200,
        'add_visits': False,
        'add_changes': False,
        'add_heatmap': False,
        'max_step': 30000,
        'render': True
        }



if __name__ == '__main__':
    infer(game, representation, experiment, infer_kwargs, **kwargs)
#   evaluate(test_params, game, representation, experiment, infer_kwargs, **kwargs)
#   analyze()
