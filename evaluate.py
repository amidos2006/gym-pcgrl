"""
Sample and graph the performance of trained models during inference experiments
with a variable parameter.
"""
import os
import shutil
import numpy as np
from matplotlib import pyplot as plt
from utils import make_vec_envs, get_exp_name, load_model

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
    '''
    Args:
        game: one of 'binary', 'sokoban', 'zelda', ...
        representation: 'narrow', 'turtle', 'wide', ...
        experiment: a name identifying the training run
    '''
    exp_name = get_exp_name(game, representation, experiment, **kwargs)
    log_dir = 'runs/{}_{}'.format(exp_name, 'log')
    model = load_model(log_dir)
    return model

def sample_data(model, env, sample_size, n_cpu, lambdas):
    sample_info = {}
    lvls = []
    for name in lambdas:
        sample_info[name] = []
    for _ in range(sample_size):
        done = np.array([False])
        obs = env.reset()
        while not done.all():
            action, _ = model.predict(obs)
            obs, rewards, done, info = env.step(action)
        lvls += [env.get_attr('pcgrl_env')[i]._rep.get_observation()['map'] for i in range(n_cpu)]
        for name in lambdas:
            sample_info[name] += [lambdas[name](info[i]) for i in range(n_cpu)]
    sample_info['diversity'] = get_hamming_diversity(lvls)
    sample_info['rewards'] = rewards
    return sample_info

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

def get_data(results, name):
    output = {}
    for n in results:
        output[n] = results[n][name]
    return output

def plt_dict(p_dict, y_title, eval_dir, file_name):
    plt.figure()
    names = []
    for name in p_dict:
        plt.plot(np.array(np.arange(0.0, 1.01, 0.1)), p_dict[name])
        names.append(name)
    plt.legend(names)
    plt.xlim(0.0, 1.0)
    plt.xticks(np.array(np.arange(0.0, 1.01, 0.1)), rotation=90)
    plt.xlabel('change percentage')
    plt.ylabel(y_title)
    plt.savefig(os.path.join(eval_dir, file_name + ".pdf"))

def analyze(problem, rep_names, exp_names, test_params, eval_name='test00', **kwargs):
    '''
    Record the final value of various environment parameters, over a certain number of trials,
    while varying some constraint on the environment (i.e., percent of map changed or
    number of steps).
    '''
    eval_name, n_cpu, sample_size = [kwargs.get(k) for k in \
        ['eval_name', 'n_cpu', 'sample_size']]
    eval_dir = "{}_{}".format(problem, eval_name)
    eval_dir = os.path.join('evals', eval_dir)
    try:
        os.mkdir(eval_dir)
    except FileExistsError:
        shutil.rmtree(eval_dir)
        os.mkdir(eval_dir)
    sample_size
    result = {}
    for i, (e_name, r_name) in enumerate(zip(exp_names, rep_names)):
        m_name = get_exp_name(problem, r_name, e_name)
        env_name = "{}-{}-v0".format(problem, r_name)
        model = get_model(problem, r_name, e_name)
        result[m_name] = {}
        for ch_perc in np.arange(0, 1.01, 0.1):
            print("Testing {} at change percentage of {}".format(m_name, ch_perc))
            kwargs['change_percentage'] = ch_perc
            env = make_vec_envs(env_name, r_name, None, n_cpu, **kwargs)
            temp_result = sample_data(model, env, sample_size, n_cpu, lambdas[problem])
            for name in temp_result:
                if not(name in result[m_name]):
                    result[m_name][name] = []
                result[m_name][name].append(np.mean(temp_result[name]))
            env.close()
            del(env)
    for param in lambdas[problem]:
        plt_dict(get_data(result, param), param, eval_dir, param)
    plt_dict(get_data(result, 'diversity'), 'diversity', eval_dir, 'diversity')

def main():
    kwargs = {
        #'change_percentage': 1,
        'target_path': 200,
        'add_visits': False,
        'add_changes': False,
        'add_heatmap': False,
        #'max_step': 30000,
        'eval_name': "with2",
        'sample_size': 100,
        'overwrite': True, # overwrite the last eval dir?
        'render': True,
        'cropped_size': 28,
        'n_cpu': 1,
        }
    problem = "binary"
    exp_names = [
        'FullyConvFix_mapOnly_1',
        'LongConv_1'
        ]
    rep_names = ['wide' for i in exp_names]
    test_params = {
        'change_percentage': [v*.1 for v in range(11)]
        }
    return analyze(problem, rep_names, exp_names, test_params, **kwargs)

if __name__ == '__main__':
    main()
