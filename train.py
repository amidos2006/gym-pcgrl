#pip install tensorflow==1.15
#Install stable-baselines as described in the documentation
from pdb import set_trace as T

import model
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from model import CustomPolicyBigMap, CApolicy, WidePolicy
from utils import get_exp_name, max_exp_idx, load_model
from envs import make_vec_envs
from stable_baselines3 import PPO
#from policy import PPO2
from stable_baselines3.common.results_plotter import load_results, ts2xy

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
    ca_action = kwargs.get('ca_action')
    map_width = kwargs.get('map_width')

    if representation == 'wide':
        if ca_action:
            policy = CApolicy
        else:
            policy = WidePolicy

        if game == "sokoban":
            T()
#           policy = FullyConvPolicySmallMap
    else:
#       policy = ActorCriticCnnPolicy
        policy = CustomPolicyBigMap

        if game == "sokoban":
            T()
#           policy = CustomPolicySmallMap
    if game == "binarygoal":
        kwargs['cropped_size'] = 32
    elif game == "zeldagoal":
        kwargs['cropped_size'] = 32
    elif game == "sokobangoal":
        kwargs['cropped_size'] = 10
    else:
        raise Exception
    n = max_exp_idx(exp_name)
    global log_dir

    if not resume:
        n = n + 1
    log_dir = 'runs/{}_{}_log'.format(exp_name, n)
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
    try:
        env, dummy_action_space, n_tools = make_vec_envs(env_name, representation, log_dir, **kwargs)
    except Exception as e:
        # if this is a new experiment, clean up the logging directory if we fail to start up
        if not resume:
            os.rmdir(log_dir)
        T()
        raise e

#       pass
    if resume:
        model = load_model(log_dir, n_tools=n_tools)



    if representation == 'wide':
        policy_kwargs = {'n_tools': n_tools}
        if ca_action:
            # FIXME: there should be a better way hahahaha
            env.action_space = dummy_action_space
            # more frequent updates, for debugging... or because our action space is huge?
            n_steps = 512
        else:
            n_steps = 2048
    else:
        policy_kwargs = {}
        # the default for SB3 PPO
        n_steps = 2048

    if not resume or model is None:
        model = PPO(policy, env, verbose=1, n_steps=n_steps, tensorboard_log="./runs", policy_kwargs=policy_kwargs)
    else:
        model.set_env(env)

   #model.policy = model.policy.to('cuda:0')
    model.policy = model.policy.cuda()
    if not logging:
        model.learn(total_timesteps=int(steps), tb_log_name=exp_name)
    else:
        model.learn(total_timesteps=int(steps), tb_log_name=exp_name, callback=callback)

from arguments import parse_args
opts = parse_args()

################################## MAIN ########################################

### User settings
conditional = True
game = opts.problem
representation = opts.representation
steps = 1e8
render = opts.render
logging = True
n_cpu = opts.n_cpu
resume = opts.resume
midep_trgs = opts.midep_trgs
ca_action = opts.ca_action
#################

max_step = 1000
global COND_METRICS
if conditional:
    experiment = 'conditional'
    COND_METRICS = opts.conditionals
    experiment = '_'.join([experiment] + COND_METRICS)
else:
    experiment = 'vanilla'
    COND_METRICS = None
    if midep_trgs:
        experiment = '_'.join([experiment, 'midepTrgs'])
if ca_action:
    experiment = '_'.join([experiment, 'CAaction'])
    max_step = 50
kwargs = {
    'map_width': 16,
    'change_percentage': 1,
    'conditional': conditional,
    'cond_metrics': COND_METRICS,
    'resume': resume,
    'max_step': max_step,
    'midep_trgs': midep_trgs,
    'ca_action': ca_action
}

if __name__ == '__main__':
    main(game, representation, experiment, steps, n_cpu, render, logging, **kwargs)
