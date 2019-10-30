
import numpy as np
from gym.utils import seeding

class Representation:
    def __init__(self, width, height, prob):
        self._width, self._height = width, height
        self._prob = prob
        total = 0
        for v in self._prob:
            self._prob[v] = max(self._prob[v], 1e-6)
        for v in self._prob:
            total += self._prob[v]
        for v in self._prob:
            self._prob[v] /= total

        self._random_start = True
        self._old_map = None

        self.seed()

    def _gen_random_map(self):
        map = np.zeros((self._height, self._width), dtype=np.uint8)
        for y in range(self._height):
            for x in range(self._width):
                total = 0
                randv = self._random.rand()
                for v in self._prob:
                    total += self._prob[v]
                    if randv < total:
                        map[y][x] = int(v)
                        break
        return map

    def seed(self, seed=None):
        self._random, seed = seeding.np_random(seed)
        return seed

    def reset(self):
        if self._random_start or self._old_map is None:
            self._map = self._gen_random_map()
            self._old_map = self._map.copy()
        else:
            self._map = self._old_map.copy()

    def adjust_param(self, **kwargs):
        self._random_start = kwargs.get('random_start', self._random_start)
        self._width, self._height = kwargs.get('width', self._width), kwargs.get('height', self._height)
        if kwargs.get('prob') is not None:
            self._prob = kwargs.get('prob')
            total = 0
            for v in self._prob:
                self._prob[v] = max(self._prob[v], 1e-6)
            for v in self._prob:
                total += self._prob[v]
            for v in self._prob:
                self._prob[v] /= total

    def get_action_space(self):
        raise NotImplementedError('get_action_space is not implemented')

    def get_observation_space(self):
        raise NotImplementedError('get_observation_space is not implemented')

    def get_observation(self):
        raise NotImplementedError('get_observation is not implemented')

    def update(self, action):
        raise NotImplementedError('update is not implemented')

    def render(self, lvl_image, tile_size, border_size):
        return lvl_image
