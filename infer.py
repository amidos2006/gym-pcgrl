"""
Run a trained agent and get generated maps
"""
import model
from model import FullyConvPolicyBigMap, FullyConvPolicySmallMap, CustomPolicyBigMap, CustomPolicySmallMap
from stable_baselines import PPO2

import time
from utils import get_exp_name, max_exp_idx, make_vec_envs, get_action

def infer(game, representation, model_path, **kwargs):
    """
     - max_trials: The number of trials per evaluation.
     - infer_kwargs: Args to pass to the environment.
    """
    env_name = '{}-{}-v0'.format(game, representation)
    if game == "binary":
        kwargs['cropped_size'] = 28
    elif game == "zelda":
        kwargs['cropped_size'] = 22
    elif game == "sokoban":
        kwargs['cropped_size'] = 10
    kwargs['render'] = True

    model = PPO2.load(model_path)
    env = make_vec_envs(env_name, representation, None, 1, **kwargs)
    obs = env.reset()
    dones = False
    final_maps = []
    for i in range(kwargs.get('trials', 1)):
        while !dones:
            action, _ = model.predict(obs)
            obs, _, dones, info = env.step(action)
            if kwargs.get('verbose', False):
                print(info[0])
            if dones:
                break
        time.sleep(0.2)

game = 'binary'
representation = 'narrow'
model_path = 'models/{}/{}/model_1.pkl'.format(game, representation)
# For inference
kwargs = {
    'change_percentage': 0.4,
    'trials': 1,
    'verbose': True
}

if __name__ == '__main__':
    infer(game, representation, model_path, **kwargs)
