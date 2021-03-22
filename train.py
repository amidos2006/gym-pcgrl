#pip install tensorflow==1.15
#Install stable-baselines as described in the documentation
from pdb import set_trace as T

import model
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from model import CustomPolicyBigMap
from utils import get_exp_name, max_exp_idx, load_model, make_vec_envs
from stable_baselines3 import PPO
#from policy import PPO2
from stable_baselines3.common.results_plotter import load_results, ts2xy

import tensorflow as tf
import numpy as np
import os

n_steps = 0
log_dir = './'
best_mean_reward, n_steps = -np.inf, 0

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO)
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
                _locals['self'].save(os.path.join(log_dir, 'best_model.zip'))
            else:
                print("Saving latest model")
                _locals['self'].save(os.path.join(log_dir, 'latest_model.zip'))
        else:
#           print('{} monitor entries'.format(len(x)))
            pass
    n_steps += 1
    # Returning False will stop training early
    return True


def main(game, representation, experiment, steps, n_cpu, render, logging, **kwargs):
    kwargs['n_cpu'] = n_cpu
    env_name = '{}-{}-v0'.format(game, representation)
    exp_name = get_exp_name(game, representation, experiment, **kwargs)
    resume = kwargs.get('resume', False)
    if representation == 'wide':
        policy = FullyConvPolicyBigMap
        if game == "sokoban":
            T()
#           policy = FullyConvPolicySmallMap
    else:
#       policy = ActorCriticCnnPolicy
        policy = CustomPolicyBigMap
        if game == "sokoban":
            T()
#           policy = CustomPolicySmallMap
#           policy = CustomPolicySmallMap
    if game == "binary":
        kwargs['cropped_size'] = 28
    elif game == "zelda":
        kwargs['cropped_size'] = 22
    elif game == "sokoban":
        kwargs['cropped_size'] = 10
    n = max_exp_idx(exp_name)
    global log_dir
    if not resume:
        n = n + 1
    log_dir = 'runs/{}_{}_{}'.format(exp_name, n, 'log')
    kwargs = {
        **kwargs,
        'render_rank': 0,
        'render': render,
    }
    used_dir = log_dir
    if not logging:
        used_dir = None
    kwargs.update({'render': render})
    if not resume:
        os.mkdir(log_dir)
    else:
        model = load_model(log_dir)

    env = make_vec_envs(env_name, representation, log_dir, **kwargs)
    if not resume or model is None:
        model = PPO(policy, env, verbose=1, tensorboard_log="./runs")
    else:
        model.set_env(env)
    if not logging:
        model.learn(total_timesteps=int(steps), tb_log_name=exp_name)
    else:
        model.learn(total_timesteps=int(steps), tb_log_name=exp_name, callback=callback)

prob_cond_metrics = {
#       'binary': ['regions'],
        'binary': ['path-length'],
        'zelda': ['num_enemies'],
        'sokoban': ['num_boxes'],
        }

################################## MAIN ########################################

### User settings
conditional = True
game = 'binary'
experiment = 'conditional'
representation = 'turtle'
steps = 1e8
render = False
logging = True
n_cpu = 50
resume = False
#################

if conditional:
    max_step = 500
    cond_metrics = prob_cond_metrics[game]
    experiment = '_'.join([experiment] + cond_metrics)
else:
    max_step = None
kwargs = {
    'conditional': conditional,
    'cond_metrics': cond_metrics,
    'resume': resume,
    'max_step': max_step,
}

if __name__ == '__main__':
    main(game, representation, experiment, steps, n_cpu, render, logging, **kwargs)
