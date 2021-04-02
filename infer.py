"""
Run a trained agent for qualitative analysis.
"""
from pdb import set_trace as T
import numpy as np
import cv2
from utils import get_exp_name, max_exp_idx, load_model, get_action
from envs import make_vec_envs


font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

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
        if EXPERIMENT_ID is None:
            n = max_exp_idx(exp_name)
        else:
            n = EXPERIMENT_ID
    if n == 0:
        raise Exception('Did not find ranked saved model of experiment: {}'.format(exp_name))
    if game == "binary":
        infer_kwargs['cropped_size'] = 32
    elif game == "zelda":
        infer_kwargs['cropped_size'] = 32
    elif game == "sokoban":
        infer_kwargs['cropped_size'] = 10
    log_dir = '{}/{}_{}_log'.format(EXPERIMENT_DIR, exp_name, n)
    # no log dir, 1 parallel environment
    n_cpu = infer_kwargs.get('n_cpu')
    env, dummy_action_space = make_vec_envs(env_name, representation, None, **infer_kwargs)
    if representation == 'wide':
        n_tools = dummy_action_space.nvec[2]
    else:
        # not true, strictly speaking, but non-wide representations don't need to know the number of tools
        n_tools = None
    model = load_model(log_dir, load_best=infer_kwargs.get('load_best'), n_tools=n_tools)
    env.action_space = dummy_action_space
    obs = env.reset()
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
    n_trials = 0
    while n_trials != max_trials:
       #action = get_action(obs, env, model)
        T()
        action, _ = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
#       print('reward: {}'.format(rewards))
#       reward = rewards[0]
#       n_regions = info[0]['regions']
#       readouts = []
#       if 'binary' in env_name:
#           curr_path_length = info[0]['path-length']
#           readouts.append('path length: {}'.format(curr_path_length) )
#           path_lengths.append(curr_path_length)
#           changes.append(info[0]['changes'])
#           regions.append(info[0]['regions'])

#       readouts += ['regions: {}'.format(n_regions), 'reward: {}'.format(reward)]
#       stringexec = ""
#       m=0
#       y0, dy = 50, 40
#       img = np.zeros((256,512,3), np.uint8)
#       scale_percent = 60 # percent of original size
#       width = int(img.shape[1] * scale_percent / 100)
#       height = int(img.shape[0] * scale_percent / 100)
#       dim = (width, height)
#       # resize image
#       for i, line in enumerate(readouts):
#           y = y0 + i*dy
#           cv2.putText(img, line, (50, y), font, fontScale, fontColor, lineType)
#          #stringexec ="cv2.putText(img, TextList[" + str(TextList.index(i))+"], (100, 100+"+str(m)+"), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 100, 100), 1, cv2.LINE_AA)\n"
#          #m += 100
#       #cv2.putText(
#       #    img,readout,
#       #    topLeftCornerOfText,
#       #    font,
#       #    fontScale,
#       #    fontColor,
#       #    lineType)
#       #Display the image
#       resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
#       cv2.imshow("img",resized)
#       cv2.waitKey(1)
#      #for p, v in model.get_parameters().items():
#      #    print(p, v.shape)
        if dones:
#          #show_state(env, path_lengths, changes, regions, n_step)
#           if 'binary' in env_name:
#               infer_info['path_lengths'] = path_lengths[-1]
#               infer_info['changes'] = changes[-1]
#               infer_info['regions'] = regions[-1]
            n_trials += 1
    return infer_info

prob_cond_metrics = {
        'binary': ['regions'],
#       'binary': ['path-length'],
        'zelda': ['num_enemies'],
        'sokoban': ['num_boxes'],
        }


from arguments import get_args
opts = get_args()

# For locating trained model
global EXPERIMENT_ID
global EXPERIMENT_DIR
#EXPERIMENT_DIR = 'hpc_runs/runs'
EXPERIMENT_DIR = 'runs'
EXPERIMENT_ID = opts.experiment_id
game = opts.problem
representation = opts.representation
conditional = True
midep_trgs = opts.midep_trgs
if conditional:
    experiment = 'conditional'
else:
    experiment = 'vanilla'
kwargs = {
       #'change_percentage': 1,
       #'target_path': 105,
       #'n': 4, # rank of saved experiment (by default, n is max possible)
        }

if conditional:
    max_step = None
    cond_metrics = opts.conditionals

    if midep_trgs:
        experiment = '_'.join(experiment, 'midepTrgs')
    experiment = '_'.join([experiment] + cond_metrics)
else:
    max_step = None
    cond_metrics = None

# For inference
infer_kwargs = {
       #'change_percentage': 1,
       #'target_path': 200,
        'conditional': True,
        'cond_metrics': cond_metrics,
        'add_visits': False,
        'add_changes': False,
        'add_heatmap': False,
        'max_step': max_step,
        'render': True,
        'n_cpu': 1,
        'load_best': opts.load_best,
        'midep_trgs': midep_trgs,
        'infer': True,
        'ca_action': opts.ca_action,
        }

if __name__ == '__main__':
    infer(game, representation, experiment, infer_kwargs, **kwargs)
#   evaluate(test_params, game, representation, experiment, infer_kwargs, **kwargs)
#   analyze()
