from gym_pcgrl.envs.reps.representation import Representation
from PIL import Image
from gym import spaces
import numpy as np

class WideRepresentation(Representation):
    def __init__(self):
        super().__init__()

    def get_action_space(self, width, height, num_tiles):
        return spaces.MultiDiscrete([width, height, num_tiles])

    def get_observation_space(self, width, height, num_tiles):
        return spaces.Box(low=0, high=num_tiles-1, dtype=np.uint8, shape=(height, width))

    def get_observation(self):
        return self._map.copy()

    def get_action_meaning(self, tiles):
        result  = "For 1st Integer:\n"
        result += "0.." + str(self._map.shape[1]-1) + ": is the x position\n"
        result += "For 2nd Integer:\n"
        result += "0.." + str(self._map.shape[0]-1) + ": is the y position\n"
        result += "For 3rd Integer:\n"
        for i in range(len(tiles)):
            result += str(i) + ": " + tiles[i] + "\n"
        return result

    def get_observation_meaning(self, tiles):
        result = "The 2D array is the current generated map where the values are:\n"
        for i in range(len(tiles)):
            result += str(i) + ": " + tiles[i] + "\n"
        return result

    def update(self, action):
        self._map[action[1]][action[0]] = action[2]
