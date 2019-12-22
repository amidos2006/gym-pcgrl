from gym_pcgrl.envs.reps.turtle_rep import TurtleRepresentation
from PIL import Image
from gym import spaces
import numpy as np
from collections import OrderedDict

"""
The turtle representation where the agent is trying to modify the position of the
turtle or the tile value of its current location similar to turtle graphics.
The difference with narrow representation is the agent now controls the next tile to be modified.
"""
class TurtleCastRepresentation(TurtleRepresentation):
    """
    Gets the action space used by the turtle representation

    Parameters:
        width: the current map width
        height: the current map height
        num_tiles: the total number of the tile values

    Returns:
        MultiDiscrete: the action space used by that turtle representation which
        correspond the movement direction + type of modification (one tile or 3x3)
        and the tile values
    """
    def get_action_space(self, width, height, num_tiles):
        return spaces.MultiDiscrete([len(self._dirs) + 2, num_tiles])

    """
    Update the turtle representation with the input action

    Parameters:
        action: an action that is used to advance the environment (same as action space)

    Returns:
        boolean: True if the action change the map, False if nothing changed
    """
    def update(self, action):
        type, value = action
        change = 0
        if type < len(self._dirs):
            self._x += self._dirs[type][0]
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
            self._y += self._dirs[type][1]
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
            type = type - len(self._dirs)
            if type == 0:
                change = [0,1][self._map[self._y][self._x] != value]
                self._map[self._y][self._x] = value
            elif type == 1:
                low_y,high_y=max(self._y-1,0),min(self._y+2,self._map.shape[0])
                low_x,high_x=max(self._x-1,0),min(self._x+2,self._map.shape[1])
                for y in range(low_y,high_y):
                    for x in range(low_x,high_x):
                        change += [0,1][self._map[y][x] != value]
                        self._map[y][x] = value
        return change, self._x, self._y
