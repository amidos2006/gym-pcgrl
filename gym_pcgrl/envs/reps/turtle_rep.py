from gym_pcgrl.envs.reps.representation import Representation
from PIL import Image
from gym import spaces
import numpy as np
from collections import OrderedDict

class TurtleRepresentation(Representation):
    def __init__(self):
        super().__init__()
        self._dirs = [(-1,0), (1,0), (0,-1), (0,1)]
        self._warp = False

    def reset(self, width, height, prob):
        super().reset(width, height, prob)
        self._x = self._random.randint(width)
        self._y = self._random.randint(height)

    def adjust_param(self, **kwargs):
        super().adjust_param(**kwargs)
        self._warp = kwargs.get('warp', self._warp)

    def get_action_space(self, width, height, num_tiles):
        return spaces.Discrete(len(self._dirs) + num_tiles)

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
        for i in range(len(self._dirs)):
            dir = ""
            if self._dirs[i][0] < 0:
                dir = "Left"
            elif self._dirs[i][0] > 0:
                dir = "Right"
            elif self._dirs[i][1] < 0:
                dir = "Up"
            elif self._dirs[i][1] > 0:
                dir = "Down"
            result += str(i) + ": Move " + dir + "\n"
        for i in range(len(tiles)):
            result += str(i+len(self._dirs)) + ": " + tiles[i] + "\n"
        return result

    def get_observation_meaning(self, tiles):
        result  = "\'pos\' is a point that identify where is the turtle at this moment\n"
        result += "\'map\' is the current generated map where the values are:\n"
        for i in range(len(tiles)):
            result += str(i) + ": " + tiles[i] + "\n"
        return result

    def update(self, action):
        if action < len(self._dirs):
            self._x += self._dirs[action][0]
            if self._x < 0:
                if self._warp:
                    self._x += self._map.shape[1]
                else:
                    self._x = 0
            if self._x >= self._map.shape[1]:
                if self._warp:
                    self._x -= self._map.shape[1]
                else:
                    self._x = self._map.shape[1] - 1
            self._y += self._dirs[action][1]
            if self._y < 0:
                if self._warp:
                    self._y += self._map.shape[0]
                else:
                    self._y = 0
            if self._y >= self._map.shape[0]:
                if self._warp:
                    self._y -= self._map.shape[0]
                else:
                    self._y = self._map.shape[0] - 1
        else:
            self._map[self._y][self._x] = action - len(self._dirs)

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
