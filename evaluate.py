
"""
Run a trained agent for qualitative analysis.
"""
import os
from pdb import set_trace as T
import numpy as np
import cv2
from utils import get_exp_name, max_exp_idx, load_model, get_action
from envs import make_vec_envs
from matplotlib import pyplot as plt
import pickle

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

def evaluate(game, representation, experiment, infer_kwargs, **kwargs):
    """
     - max_trials: The number of trials per evaluation.
     - infer_kwargs: Args to pass to the environment.
    """
    infer_kwargs = {
            **infer_kwargs,
            'inference': True,
            'evaluate': True
            }
    max_trials = kwargs.get('max_trials', -1)
    n = kwargs.get('n', None)
    map_width = infer_kwargs.get('map_width')
    max_steps = infer_kwargs.get('max_steps')
    eval_controls = infer_kwargs.get('eval_controls')
    env_name = '{}-{}-v0'.format(game, representation)
    exp_name = get_exp_name(game, representation, experiment, **kwargs)
    if n is None:
        if EXPERIMENT_ID is None:
            n = max_exp_idx(exp_name)
        else:
            n = EXPERIMENT_ID
    if n == 0:
        raise Exception('Did not find ranked saved model of experiment: {}'.format(exp_name))
    crop_size = infer_kwargs.get('cropped_size')
    if crop_size is None:
        if game == "binarygoal":
            infer_kwargs['cropped_size'] = 32
        elif game == "zeldagoal":
            infer_kwargs['cropped_size'] = 28
        elif game == "sokobangoal":
            infer_kwargs['cropped_size'] = 10
    log_dir = '{}/{}_{}_log'.format(EXPERIMENT_DIR, exp_name, n)
    data_path = os.path.join(log_dir, '{}_eval_data.pkl'.format(eval_controls))
    if VIS_ONLY:
        eval_data = pickle.load(open(data_path, "rb"))
        visualize_data(eval_data, log_dir)
        return
    # no log dir, 1 parallel environment
    n_cpu = infer_kwargs.get('n_cpu')
    env, dummy_action_space, n_tools = make_vec_envs(env_name, representation, None, **infer_kwargs)
    model = load_model(log_dir, load_best=infer_kwargs.get('load_best'), n_tools=n_tools)
#   model.set_env(env)
    env.action_space = dummy_action_space
    # Record final values of each trial
#   if 'binary' in env_name:
#       path_lengths = []
#       changes = []
#       regions = []
#       infer_info = {
#           'path_lengths': [],
#           'changes': [],
#           'regions': [],
#           }
    if n_cpu == 1:
        control_bounds = env.envs[0].get_control_bounds()
    elif n_cpu > 1:
        env.remotes[0].send(('env_method', ('get_control_bounds', [], {})))  # supply args and kwargs
        control_bounds = env.remotes[0].recv()
    if not eval_controls:
        eval_controls = control_bounds.keys()
    ctrl_bounds = [(k, control_bounds[k]) for k in eval_controls]
    # Hackish get initial states
    init_states = []
    for i in range(N_TRIALS):
        env.envs[0].reset()
        init_states.append(env.envs[0].unwrapped._rep._map)
    if len(ctrl_bounds) == 1:
        step_size = STEP_0
        ctrl_name = ctrl_bounds[0][0]
        bounds = ctrl_bounds[0][1] 
        eval_trgs = np.arange(bounds[0], bounds[1] + 1, step_size)
        level_images = []
        cell_scores = np.zeros((len(eval_trgs), 1))
        cell_ctrl_scores = np.zeros(shape=(len(eval_trgs), 1))
        cell_static_scores = np.zeros(shape=(len(eval_trgs), 1))
        N_EVALS = N_TRIALS * N_MAPS
        for i, trg in enumerate(eval_trgs):
            trg_dict = {ctrl_name: trg}
            print('evaluating control targets: {}'.format(trg_dict))
            env.envs[0].set_trgs(trg_dict)
#           set_ctrl_trgs(env, {ctrl_name: trg})
            net_score, ctrl_score, static_score, level_image = eval_episodes(model, env, N_EVALS, n_cpu, init_states)
            level_images.append(level_image)
            cell_scores[i] = net_score
            cell_ctrl_scores[i] = ctrl_score
            cell_static_scores[i] = static_score
        if RENDER_LEVELS:
            T()
        ctrl_names = (ctrl_name, None)
        ctrl_ranges = (eval_trgs, None)
    elif len(ctrl_bounds) >=2:
        step_0 = STEP_0
        step_1 = STEP_1
        ctrl_0, ctrl_1 = ctrl_bounds[0][0], ctrl_bounds[1][0]
        b0, b1 = ctrl_bounds[0][1], ctrl_bounds[1][1]
        trgs_0 = np.arange(b0[0], b0[1]+0.5, step_0)
        trgs_1 = np.arange(b1[0], b1[1]+0.5, step_1)
        cell_scores = np.zeros(shape=(len(trgs_0), len(trgs_1)))
        cell_ctrl_scores = np.zeros(shape=(len(trgs_0), len(trgs_1)))
        cell_static_scores = np.zeros(shape=(len(trgs_0), len(trgs_1)))
        trg_dict = env.envs[0].static_trgs
        trg_dict = dict([(k, min(v)) if isinstance(v, tuple) else (k, v) for (k, v) in trg_dict.items()])
        for i, t0 in enumerate(trgs_0):
            for j, t1 in enumerate(trgs_1):
                ctrl_trg_dict = {ctrl_0: t0, ctrl_1: t1}
                trg_dict.update(ctrl_trg_dict)
                print('evaluating control targets: {}'.format(trg_dict))
                env.envs[0].set_trgs(trg_dict)
    #           set_ctrl_trgs(env, {ctrl_name: trg})
                net_score, ctrl_score, static_score = eval_episodes(model, env, N_EVALS, n_cpu, init_states)
                cell_scores[i, j] = net_score
                cell_ctrl_scores[i, j] = ctrl_score
                cell_static_scores[i, j] = static_score
        ctrl_names = (ctrl_0, ctrl_1)
        ctrl_ranges = (trgs_0, trgs_1)

    eval_data = EvalData(ctrl_names, ctrl_ranges, cell_scores, cell_ctrl_scores, cell_static_scores)
    pickle.dump(eval_data, open(data_path, "wb"))
    visualize_data(eval_data, log_dir)


def eval_episodes(model, env, n_trials, n_envs, init_states):
    eval_scores = np.zeros(n_trials)
    eval_ctrl_scores = np.zeros(n_trials)
    eval_static_scores = np.zeros(n_trials)
    n = 0
    # FIXME: why do we need this?
    while n < n_trials:
        env.envs[0].set_map(init_states[n % N_MAPS])
        obs = env.reset()
#       epi_rewards = np.zeros((max_step, n_envs))
        i = 0
        # note that this is weighted loss
        init_loss = env.envs[0].get_loss()
        init_ctrl_loss = env.envs[0].get_ctrl_loss()
        init_static_loss = env.envs[0].get_static_loss()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, rewards, done, info = env.step(action)
#           epi_rewards[i] = rewards
            i += 1
        final_loss = env.envs[0].get_loss()
        final_ctrl_loss = env.envs[0].get_ctrl_loss()
        final_static_loss = env.envs[0].get_static_loss()
        # what percentage of loss (distance from target) was recovered?
        eps = 0.001
        max_loss = max(abs(init_loss), eps)
        max_ctrl_loss = max(abs(init_ctrl_loss), eps)
        max_static_loss = max(abs(init_static_loss), eps)
        score = (final_loss - init_loss) / abs(max_loss)
        ctrl_score = (final_ctrl_loss - init_ctrl_loss) / abs(max_ctrl_loss)
        static_score = (final_static_loss - init_static_loss) / abs(max_static_loss)
        eval_scores[n] = score
        eval_ctrl_scores[n] = ctrl_score
        eval_static_scores[n] = static_score
        n += n_envs
    eval_score = eval_scores.mean()
    eval_ctrl_score = eval_ctrl_scores.mean()
    eval_static_score = eval_static_scores.mean()
    print('eval score: {}'.format(eval_score))
    print('control score: {}'.format(ctrl_score))
    print('static score: {}'.format(static_score))
    if RENDER_LEVELS:
        level_image = env.envs[0].render('rgb_array')
        print(level_image)
        print(level_image.shape)
    else:
        level_image = None
    return eval_score, eval_ctrl_score, eval_static_score, level_image


def visualize_data(eval_data, log_dir):

    def create_heatmap(title, data):
        fig, ax = plt.subplots()
        # percentages from ratios
        data = data * 100
        data = np.clip(data, -200, 100)
        data = data.T
        if data.shape[0] == 1:
            fig.set_size_inches(10, 2)
            ax.set_yticks([])
            tick_idxs = np.arange(0, cell_scores.shape[0], 10)
            ticks = np.arange(cell_scores.shape[0])
            ticks = ticks[tick_idxs]
            ax.set_xticks(ticks)
            labels = np.array([int(x) if x % 50 == 0 else "" for (i, x) in enumerate(ctrl_ranges[0])])
            labels = labels[tick_idxs]
            ax.set_xticklabels(labels)
        else:
            ax.set_xticks(np.arange(cell_scores.shape[0]))
            ax.set_yticks(np.arange(cell_scores.shape[1]))
            ax.set_xticklabels([int(x) for x in ctrl_ranges[0]])
            ax.set_yticklabels([int(x) for x in ctrl_ranges[1]])
        # Create the heatmap
        im = ax.imshow(data, aspect='auto')

        #Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("", rotation=90, va="bottom")

        # We want to show all ticks...
        # ... and label them with the respective list entries
        plt.xlabel(ctrl_names[0])
        plt.ylabel(ctrl_names[1])

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        ax.set_title(title)
        fig.tight_layout()

        plt.savefig(os.path.join(log_dir, "{}_{}.png".format(ctrl_names, title)))
        plt.show()

    ctrl_names = eval_data.ctrl_names
    ctrl_ranges = eval_data.ctrl_ranges
    cell_scores = eval_data.cell_scores
    cell_ctrl_scores = eval_data.cell_ctrl_scores
    cell_static_scores = eval_data.cell_static_scores

    title = "All goals (mean progress, %)"
    create_heatmap(title, cell_scores)

    title = "Controlled goals (mean progress, %)"
    create_heatmap(title, cell_ctrl_scores)

    title = "Fixed goals (mean progress, %)"
    create_heatmap(title, cell_static_scores)



class EvalData():
    def __init__(self, ctrl_names, ctrl_ranges, cell_scores, cell_ctrl_scores, cell_static_scores):
        self.ctrl_names = ctrl_names
        self.ctrl_ranges = ctrl_ranges
        self.cell_scores = cell_scores
        self.cell_ctrl_scores = cell_ctrl_scores
        self.cell_static_scores = cell_static_scores



#NOTE: let's not try multiproc how about that :~)

#def eval_episodes(model, env, n_trials, n_envs):
#    eval_scores = np.zeros(n_trials)
#    n = 0
#    # FIXME: why do we need this?
#    env.reset()
#    while n < n_trials:
#
#        obs = env.reset()
##       epi_rewards = np.zeros((max_step, n_envs))
#        i = 0
##       env.remotes[0].send(('env_method', ('get_metric_vals', [], {})))  # supply args and kwargs
##       init_metric_vals = env.remotes[0].recv()
#        [remote.send(('env_method', ('get_loss', [], {}))) for remote in env.remotes]
#        # note that this is weighted loss
#        init_loss = np.sum([remote.recv() for remote in env.remotes])
#        dones = np.array([False])
#        while not dones.all():
#            action, _ = model.predict(obs)
#            obs, rewards, dones, info = env.step(action)
##           epi_rewards[i] = rewards
#            i += 1
#        # since reward is weighted loss
#        final_loss = np.sum(rewards)
#        # what percentage of loss (distance from target) was recovered?
#        score = (final_loss - init_loss) / abs(init_loss)
##       env.remotes[0].send(('env_method', ('get_metric_vals', [], {})))  # supply args and kwargs
##       final_metric_vals = env.remotes[0].recv()
#        eval_scores[n] = score
#        n += n_envs
#    return eval_scores.mean()
#
#def set_ctrl_trgs(env, trg_dict):
#    [remote.send(('env_method', ('set_trgs', [trg_dict], {}))) for remote in env.remotes]

from arguments import get_args
args = get_args()
args.add_argument('--vis_only',
        help='Just load data from previous evaluation and visualize it.',
        action='store_true',
        )
args.add_argument('--eval_controls',
        help='Which controls to evaluate and visualize.',
        nargs='+',
        default=[],
        )
args.add_argument('--n_maps',
        help='Number maps on which to simulate in each cell.',
        default=3,
        type=int,
        )
args.add_argument('--n_trials',
        help='Number trials for which to simulate on each map.',
        default=3,
        type=int,
        )
args.add_argument('--step_size',
        help='Bin size along either dimension.',
        default=20,
        type=int,
        )
args.add_argument('--render_levels',
        help='Save final maps (default to only one eval per cell)',
        action='store_true',
        )
opts = args.parse_args()
global VIS_ONLY 
VIS_ONLY = opts.vis_only

# For locating trained model
global EXPERIMENT_ID
global EXPERIMENT_DIR
#EXPERIMENT_DIR = 'hpc_runs/runs'
if not opts.HPC:
    EXPERIMENT_DIR = 'runs'
else:
    EXPERIMENT_DIR = 'hpc_runs'
EXPERIMENT_ID = opts.experiment_id
problem = opts.problem
representation = opts.representation
conditional = True
midep_trgs = opts.midep_trgs
ca_action = opts.ca_action
if conditional:
    experiment = 'conditional'
else:
    experiment = 'vanilla'
kwargs = {
       #'change_percentage': 1,
       #'target_path': 105,
       #'n': 4, # rank of saved experiment (by default, n is max possible)
        }

if problem == 'sokoban':
    map_width = 5
else:
    map_width = 16

if conditional:
    max_step = 1000
    cond_metrics = opts.conditionals

    experiment = '_'.join([experiment] + cond_metrics)
    if midep_trgs:
        experiment = '_'.join([experiment, 'midepTrgs'])
    if ca_action:
        max_step = 50
        experiment = '_'.join([experiment, 'CAaction'])
else:
    max_step = None
    cond_metrics = None

# For inference
infer_kwargs = {
       #'change_percentage': 1,
       #'target_path': 200,
        'conditional': True,
        'cond_metrics': cond_metrics,
        'max_step': max_step,
        'render': opts.render,
        # TODO: multiprocessing
#       'n_cpu': opts.n_cpu,
        'n_cpu': 1,
        'load_best': opts.load_best,
        'midep_trgs': midep_trgs,
        'infer': True,
        'ca_action': ca_action,
        'map_width': map_width,
        'eval_controls': opts.eval_controls,
        'cropped_size': opts.crop_size,
        }

global STEP_0
global STEP_1
STEP_0 = opts.step_size
STEP_1 = opts.step_size
RENDER_LEVELS = opts.render_levels

if RENDER_LEVELS:
    N_MAPS = 1
    N_TRIALS = 1
else:
    N_MAPS = opts.n_maps
    N_TRIALS = opts.n_trials

if __name__ == '__main__':

    evaluate(problem, representation, experiment, infer_kwargs, **kwargs)
#   evaluate(test_params, game, representation, experiment, infer_kwargs, **kwargs)
#   analyze()
