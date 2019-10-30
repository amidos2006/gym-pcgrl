from gym_pcgrl.envs.reps.representation import Representation
from PIL import Image
from gym import spaces
import numpy as np
from collections import OrderedDict

class TurtleRepresentation(Representation):
    def __init__(self, width, height, prob):
        super().__init__(width, height, prob)
        self._dirs = [(-1,0), (1,0), (0,-1), (0,1)]

    def reset(self):
        super().reset()
        self._x = self._random.randint(self._width)
        self._y = self._random.randint(self._height)

    def get_action_space(self):
        return spaces.Discrete(len(self._dirs) + len(self._prob))

    def get_observation_space(self):
        return spaces.Dict({
            "pos": spaces.Box(low=np.array([0, 0]), high=np.array([self._width-1, self._height-1]), dtype=np.uint8),
            "map": spaces.Box(low=0, high=len(self._prob)-1, dtype=np.uint8, shape=(self._height, self._width))
        })

    def get_observation(self):
        return OrderedDict({
            "pos": np.array([self._x, self._y], dtype=np.uint8),
            "map": self._map.copy()
        })

    def update(self, action):
        if action < len(self._dirs):
            self._x += self._dirs[action][0]
            if self._x < 0:
                self._x = 0
            if self._x >= self._width:
                self._x = self._width - 1
            self._y += self._dirs[action][1]
            if self._y < 0:
                self._y = 0
            if self._y >= self._height:
                self._y = self._height - 1
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
