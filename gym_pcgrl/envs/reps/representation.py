
import numpy as np
from gym.utils import seeding

class Representation:
    def __init__(self):
        self._random_start = True
        self._map = None
        self._old_map = None

        self.seed()

    def _gen_random_map(self, width, height, prob):
        map = np.zeros((height, width), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                total = 0
                randv = self._random.rand()
                for v in prob:
                    total += prob[v]
                    if randv < total:
                        map[y][x] = int(v)
                        break
        return map

    def seed(self, seed=None):
        self._random, seed = seeding.np_random(seed)
        return seed

    def reset(self, width, height, prob):
        if self._random_start or self._old_map is None:
            self._map = self._gen_random_map(width, height, prob)
            self._old_map = self._map.copy()
        else:
            self._map = self._old_map.copy()

    def adjust_param(self, **kwargs):
        self._random_start = kwargs.get('random_start', self._random_start)

    def get_action_space(self, width, height, num_tiles):
        raise NotImplementedError('get_action_space is not implemented')

    def get_observation_space(self, width, height, num_tiles):
        raise NotImplementedError('get_observation_space is not implemented')

    def get_action_meaning(self, tiles):
        raise NotImplementedError('get_action_meaning is not implemented')

    def get_observation_meaning(self, tiles):
        raise NotImplementedError('get_observation_meaning is not implemented')

    def get_observation(self):
        raise NotImplementedError('get_observation is not implemented')

    def update(self, action):
        raise NotImplementedError('update is not implemented')

    def render(self, lvl_image, tile_size, border_size):
        return lvl_image
