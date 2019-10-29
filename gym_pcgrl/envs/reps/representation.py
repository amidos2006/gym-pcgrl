
from PIL import Image
import numpy as np
from gym.utils import seeding

class Representation:
    def __init__(self):
        self._width = -1
        self._height = -1
        self._prob = None
        self._random_start = False
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

    def _init_param(self, width, height, prob):
        self._random_start = True
        self._width, self._height = width, height
        self._prob = prob
        total = 0
        for v in self._prob:
            self._prob[v] = max(self._prob[v], 1e-6)
        for v in self._prob:
            total += self._prob[v]
        for v in self._prob:
            self._prob[v] /= total

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

    def update(self, action):
        pass

    def render(self, graphics, padding_tile, tile_size):
        lvlImage = Image.new("RGBA", ((self._width+2)*tile_size, (self._height+2)*tile_size), (0,0,0,255))
        for y in range(self._height+2):
            lvlImage.paste(graphics[str(padding_tile)], (0, y*tile_size, tile_size, (y+1)*tile_size))
            lvlImage.paste(graphics[str(padding_tile)], ((self._width+1)*tile_size, y*tile_size, (self._width+2)*tile_size, (y+1)*tile_size))
        for x in range(self._width+2):
            lvlImage.paste(graphics[str(padding_tile)], (x*tile_size, 0, (x+1)*tile_size, tile_size))
            lvlImage.paste(graphics[str(padding_tile)], (x*tile_size, (self._height+1)*tile_size, (x+1)*tile_size, (self._height+2)*tile_size))
        for y in range(self._height):
            for x in range(self._width):
                lvlImage.paste(graphics[str(int(self._map[y][x]))], ((x+1)*tile_size, (y+1)*tile_size, (x+2)*tile_size, (y+2)*tile_size))
        return lvlImage
