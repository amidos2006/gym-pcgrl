from gym_pcgrl.envs.reps.representation import Representation
from PIL import Image
from gym import spaces
import numpy as np
from collections import OrderedDict

class NarrowRepresentation(Representation):
    def __init__(self):
        super().__init__()
        self._random_tile = True

    def reset(self, width, height, prob):
        super().reset(width, height, prob)
        self._x = 0
        self._y = 0
        if self._random_tile:
            self._x = self._random.randint(width)
            self._y = self._random.randint(height)

    def get_action_space(self, width, height, num_tiles):
        return spaces.Discrete(num_tiles)

    def get_observation_space(self, width, height, num_tiles):
        return spaces.Dict({
            "pos": spaces.Box(low=np.array([0, 0]), high=np.array([width-1, height-1]), dtype=np.uint8),
            "map": spaces.Box(low=0, high=num_tiles-1, dtype=np.uint8, shape=(height, width))
        })

    def get_observation(self):
        return OrderedDict({
            "pos": np.array([self._x, self._y], dtype=np.uint8),
            "map": self._map.copy()
        })

    def get_action_meaning(self, tiles):
        result = ""
        for i in range(len(tiles)):
            result += str(i) + ": " + tiles[i] + "\n"
        return result

    def get_observation_meaning(self, tiles):
        result  = "\'pos\' is a point that identify which tile is going to be modified\n"
        result += "\'map\' is the current generated map where the values are:\n"
        for i in range(len(tiles)):
            result += str(i) + ": " + tiles[i] + "\n"
        return result

    def adjust_param(self, **kwargs):
        super().adjust_param(**kwargs)
        self._random_tile = kwargs.get('random_tile', self._random_tile)

    def update(self, action):
        self._map[self._y][self._x] = action
        if self._random_tile:
            self._x = self._random.randint(self._map.shape[1])
            self._y = self._random.randint(self._map.shape[0])
        else:
            self._x += 1
            if self._x >= self._map.shape[1]:
                self._x = 0
                self._y += 1
                if self._y >= self._map.shape[0]:
                    self._y = 0

    def render(self, lvl_image, tile_size, border_size):
        x_graphics = Image.new("RGBA", (tile_size,tile_size), (0,0,0,0))
        for x in range(tile_size):
            x_graphics.putpixel((0,x),(255,0,0,255))
            x_graphics.putpixel((1,x),(255,0,0,255))
            x_graphics.putpixel((tile_size-2,x),(255,0,0,255))
            x_graphics.putpixel((tile_size-1,x),(255,0,0,255))
        for y in range(tile_size):
            x_graphics.putpixel((y,0),(255,0,0,255))
            x_graphics.putpixel((y,1),(255,0,0,255))
            x_graphics.putpixel((y,tile_size-2),(255,0,0,255))
            x_graphics.putpixel((y,tile_size-1),(255,0,0,255))
        lvl_image.paste(x_graphics, ((self._x+border_size)*tile_size, (self._y+border_size)*tile_size,
                                        (self._x+border_size+1)*tile_size,(self._y+border_size+1)*tile_size), x_graphics)
        return lvl_image
