from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from train import get_exp_name, max_exp_idx, load_model, make_env
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


def infer(game, representation, experiment, infer_kwargs, **kwargs):
    infer_kwargs = {
            **infer_kwargs,
            'inference': True,
            'render': True,
            }
    env_name = '{}-{}-v0'.format(game, representation)
    exp_name = get_exp_name(game, representation, experiment, **kwargs)
    if n is None:
        n = max_exp_idx(exp_name)
    if n == 0:
        raise Exception('Did not find ranked saved model of experiment: {}'.format(exp_name))
    log_dir = 'runs/{}_{}_{}'.format(exp_name, n, 'log')
    model = load_model(log_dir)
    log_dir = None
   #log_dir = os.path.join(log_dir, 'eval')
    kwargs = {
            **kwargs,
            'change_percentage': 1,
            'target_path': 98,
            }
    env = DummyVecEnv([make_env(env_name, representation, 0, log_dir, **infer_kwargs)])
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
               #show_state(env, path_length, changes, regions, n_step)
                pass
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


# For locating trained model
game = 'binary'
representation = 'wide'
experiment = None
kwargs = {
        'change_percentage': 0.2,
        'target_path': 48,
        'render': True,
        'n': 5, # rank of saved experiment
        }

# For inference
infer_kwargs = {
        'change_percentage': 1,
        'target_path': 200,
        'render': True,
        }


if __name__ == '__main__':
    infer(game, representation, experiment, infer_kwargs, **kwargs)
