"""
Sample and graph the performance of trained models during inference experiments
with a variable parameter.
"""
import os
import shutil
import numpy as np
from utils import make_vec_envs
from utils import get_exp_name, load_model


binary_lambdas = {
    'pathlength': lambda info: info['path-length'],
    'regions': lambda info: info['regions'],
    'iterations': lambda info: info['iterations'] / (1.0 * info['max_iterations']),
    'changes': lambda info: info['changes'] / (1.0 * info['max_changes']),
    'pathlength_const': lambda info: [0, 1][info['path-length'] >= 48],
    'regions_const': lambda info: [0, 1][info['regions'] == 1],
}
zelda_lambdas = {
    'player': lambda info: info['player'],
    'key': lambda info: info['key'],
    'door': lambda info: info['door'],
    'regions': lambda info: info['regions'],
    'nearestenemy': lambda info: info['nearest-enemy'],
    'pathlength': lambda info: info['path-length'],
    'iterations': lambda info: info['iterations'] / (1.0 * info['max_iterations']),
    'changes': lambda info: info['changes'] / (1.0 * info['max_changes']),
    'player_const': lambda info: [0, 1][info['player'] == 1],
    'key_const': lambda info: [0, 1][info['key'] == 1],
    'door_const': lambda info: [0, 1][info['door'] == 1],
    'regions_const': lambda info: [0, 1][info['regions'] == 1],
    'nearestenemy_const': lambda info: [0, 1][info['nearest-enemy'] >= 4],
    'pathlength_const': lambda info: [0, 1][info['path-length'] >= 16],
}
sokoban_lambdas = {
    'player': lambda info: info['player'],
    'crate': lambda info: info['crate'],
    'target': lambda info: info['target'],
    'regions': lambda info: info['regions'],
    'sollength': lambda info: info['sol-length'],
    'iterations': lambda info: info['iterations'] / (1.0 * info['max_iterations']),
    'changes': lambda info: info['changes'] / (1.0 * info['max_changes']),
    'player_const': lambda info: [0, 1][info['player'] == 1],
    'ratio_const': lambda info: [0, 1][info['crate'] == info['target'] and info['crate'] > 0],
    'sollength_const': lambda info: [0, 1][info['sol-length'] >= 18],
}
lambdas = {
    'binary': binary_lambdas,
    'zelda': zelda_lambdas,
    'sokoban': sokoban_lambdas
}

def get_model(game, representation, experiment, **kwargs):
    exp_name = get_exp_name(game, representation, experiment, **kwargs)
    log_dir = 'runs/{}_{}'.format(exp_name, 'log')
    model = load_model(log_dir)
    return model

def sample_data(model, sample_size, env, lambdas):
    sample_info = {}
    lvls = []
    for name in lambdas:
        sample_info[name] = []
    for i in range(sample_size):
        done = np.array([False])
        obs = env.reset()
        if done.all():
            action, _ = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            lvls.append(env.get_attr('pcgrl_env')[0]._rep.get_observation()['map'])
            for name in lambdas:
                sample_info[name].append(lambdas[name](info[0]))
    sample_info['diversity'] = get_hamming_diversity(lvls)
    return sample_info

def analyze():
    try:
        os.mkdir(eval_dir)
    except FileExistsError:
        shutil.rmtree(eval_dir)
        os.mkdir(eval_dir)
    result = {}
    for i in range(len(exp_names)):
        r_name = rep_names[i]
        e_name = exp_names[i]
        m_name = get_exp_name(problem, r_name, e_name)
        env_name = "{}-{}-v0".format(problem, r_name)
        model = get_model(problem, r_name, e_name)
        result[m_name] = {}
        for ch_perc in np.arange(0, 1.01, 0.1):
            print("Testing {} at change percentage of {}".format(m_name, ch_perc))
            kwargs['change_percentage'] = ch_perc
            env = make_vec_envs(env_name, r_name, None, n_cpu, **infer_kwargs)
            temp_result = sample_data(model, sample_size, env, lambdas[problem])
            for name in temp_result:
                if not(name in result[m_name]):
                    result[m_name][name] = []
                result[m_name][name].append(np.mean(temp_result[name]))
            env.close()
            del(env)
    for n in lambdas[problem]:
        plt_dict(get_data(result, n), n, n)
    plt_dict(get_data(result, 'diversity'), 'diversity', 'diversity')

def get_hamming_diversity(lvls):

    hamming = []
    for i in range(len(lvls)):
        lvl1 = lvls[i]
        lvl_hamming = []
        for j in range(len(lvls)):
            lvl2 = lvls[j]
            if i != j:
                diff = np.clip(abs(lvl1 - lvl2), 0, 1)
                lvl_hamming.append(diff.sum())
        hamming.append(np.mean(lvl_hamming) / (lvls[0].shape[0] * lvls[0].shape[1]))
    return hamming

def show_state(env, l, c, r, step=0, name="", info=""):
    fig = plt.figure(10)
    plt.clf()
    plt.title("{} | Step: {} Path: {} Changes: {} Regions: {}".format(name, step, l[-1], c[-1], r[-1]))
    ax1 = fig.add_subplot(1,4,1)
    ax1 = plt.imshow(env.render(mode='rgb_array'))
    plt.axis('off')
   #ax2 = fig.add_subplot(1,4,2)
   #ax2 = plt.plot(l)
   #ax3 = fig.add_subplot(1,4,3)
   #ax3 = plt.plot(c)
   #ax4 = fig.add_subplot(1,4,4)
   #ax4 = plt.plot(r)
   #fig.set_figwidth(15)
   #plt.tight_layout()
   #plt.show()
    display.clear_output(wait=True)
    display.display(plt.gcf())

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
def get_data(results, name):
    output = {}
    for n in results:
        output[n] = results[n][name]
    return output

def plt_dict(p_dict, y_title, file_name):
    plt.figure()
    names = []
    for name in p_dict:
        plt.plot(np.array(np.arange(0.0,1.01,0.1)),p_dict[name])
        names.append(name)
    plt.legend(names)
    plt.xlim(0.0,1.0)
    plt.xticks(np.array(np.arange(0.0,1.01,0.1)), rotation=90)
    plt.xlabel('change percentage')
    plt.ylabel(y_title)
    plt.savefig(os.path.join(eval_dir, file_name + ".pdf"))

# For inference
infer_kwargs = {
       #'change_percentage': 1,
        'target_path': 200,
        'add_visits': False,
        'add_changes': False,
        'add_heatmap': False,
       #'max_step': 30000,
        'render': True
        }
test_params = {
        'change_percentage': [v*.1 for v in range(11)]
        }

problem = "binary"
eval_name = "with2"
eval_name = "{}_{}".format(problem, eval_name)
eval_dir = os.path.join('evals', eval_name)
sample_size = 100
overwrite = True # overwrite the last eval dir?
exp_names = [
        'FullyConvFix_mapOnly_1',
        'LongConv_1'
        ]
rep_names = ['wide' for i in exp_names]
kwargs={
    'cropped_size': 28,
}
n_cpu = 1

if __name__ == '__main__':
    analyze()
