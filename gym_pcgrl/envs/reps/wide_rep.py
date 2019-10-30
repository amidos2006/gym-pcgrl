from gym_pcgrl.envs.reps.representation import Representation
from PIL import Image
from gym import spaces
import numpy as np

class WideRepresentation(Representation):
    def __init__(self, width, height, prob):
        super().__init__(width, height, prob)

    def get_action_space(self):
        return spaces.MultiDiscrete([self._width, self._height, len(self._prob)])

    def get_observation_space(self):
        return spaces.Box(low=0, high=len(self._prob)-1, dtype=np.uint8, shape=(self._height, self._width))

    def get_observation(self):
        return self._map.copy()

    def update(self, action):
        self._map[action[1]][action[0]] = action[2]
