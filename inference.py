from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from helper import get_exp_name, max_exp_idx, load_model, make_env
import matplotlib.pyplot as plt
import numpy as np


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


def infer(game, representation, experiment, max_trials, infer_kwargs, **kwargs):
    '''
     - max_trials: The number of trials per evaluation.
     - infer_kwargs: Args to pass to the environment.
    '''
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
    kwargs = {
            **kwargs,
            'change_percentage': 1,
            'target_path': 98,
            }
    env = DummyVecEnv([make_env(env_name, representation, 0, None, **infer_kwargs)])
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
    max_trials = max_trials
    n_trials = 0
    while n_trials != max_trials:
        action = get_action(obs, env, model)
        obs, rewards, dones, info = env.step(action)
        if 'binary' in env_name:
            path_length.append(info[0]['path-length'])
            changes.append(info[0]['changes'])
            regions.append(info[0]['regions'])
        if dones:
           #show_state(env, path_length, changes, regions, n_step)
            if 'binary' in env_name:
                infer_info['path_length'] = path_length[-1]
                infer_info['changes'] = changes[-1]
                infer_info['regions'] = regions[-1]
            n_trials += 1
    return infer_info


def evaluate(test_params, *args, **kwargs):
    '''
    - test_params: A dictionary mapping parameters of the environment to lists of values
                  to be tested. Must apply to the environment specified by args.
    '''
    eval_info = {}
    # set environment parameters
    for param, val in test_params.items():
        kwargs[param] = val
        infer_info = infer(*args, **kwargs)
        # get average of metrics over trials
        for k, v in infer_info.items():
            N = len(v)
            mean = sum(v) / N
            stdev = (sum([(mean - v_i) ** 2 for v_i in v]) / (N - 1)) ** .5
            eval_info[k] = (mean, stdev)
            print(eval_info)


# For locating trained model
game = 'binary'
representation = 'wide'
experiment = 'FullyConv2'
kwargs = {
        'max_trials': -1,
        'change_percentage': 1,
        'target_path': 105,
        'render': True,
       #'n': 1, # rank of saved experiment
        }

# For inference
infer_kwargs = {
       #'change_percentage': 1,
       #'target_path': 200,
        'max_step': 1500,
        'render': True,
        }

test_params = {
        'change_percentage': [range(11)/10]
        }


if __name__ == '__main__':
#   infer(game, representation, experiment, infer_kwargs, **kwargs)
    evaluate(game, representation, experiment, infer_kwargs, **kwargs)
