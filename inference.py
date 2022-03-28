"""
Run a trained agent and get generated maps
"""
import model
from stable_baselines import PPO2

import time
from utils import make_vec_envs
import numpy as np
import cv2

def save_map(obs):
    empty_tile = cv2.imread(f"./gym_pcgrl/envs/probs/binary/empty.png")
    solid_tile = cv2.imread(f"./gym_pcgrl/envs/probs/binary/solid.png")
    map = np.zeros((empty_tile.shape[0]*obs.shape[0], empty_tile.shape[1]*obs.shape[1], 3))

    for i in range(obs.shape[0]):
        for j in range(obs.shape[1]):
            if obs[i][j] == 0:
                map[i][j] = empty_tile
            else:
                map[i][j] = solid_tile

    cv2.imwrite(f"./inference/{representation}_{1}.png",map)

def infer(game, representation, model_path, **kwargs):
    """
     - max_trials: The number of trials per evaluation.
     - infer_kwargs: Args to pass to the environment.
    """
    env_name = '{}-{}-v0'.format(game, representation)
    if game == "binary":
        model.FullyConvPolicy = model.FullyConvPolicyBigMap
        kwargs['cropped_size'] = 28
    elif game == "zelda":
        model.FullyConvPolicy = model.FullyConvPolicyBigMap
        kwargs['cropped_size'] = 22
    elif game == "sokoban":
        model.FullyConvPolicy = model.FullyConvPolicySmallMap
        kwargs['cropped_size'] = 10
    kwargs['render'] = True

    agent = PPO2.load(model_path)
    env = make_vec_envs(env_name, representation, None, 1, **kwargs)
    obs = env.reset()
    obs = env.reset()
    dones = False
    for i in range(kwargs.get('trials', 1)):
        while not dones:
            action, _ = agent.predict(obs)
            obs, _, dones, info = env.step(action)
            if kwargs.get('verbose', False):
                print(info[0])
            if dones:
                break
        time.sleep(0.2)



################################## MAIN ########################################
game = 'pp'
representation = 'turtle'
model_path = 'models/{}/{}/model_1.pkl'.format(game, representation)
kwargs = {
    'change_percentage': 0.4,
    'trials': 1,
    'verbose': True
}

if __name__ == '__main__':
    infer(game, representation, model_path, **kwargs)
