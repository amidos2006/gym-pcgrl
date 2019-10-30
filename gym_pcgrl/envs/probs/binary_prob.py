import os
from PIL import Image
from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.probs.helper import calc_num_regions, calc_longest_path

class BinaryProblem(Problem):
    def __init__(self):
        super().__init__()
        
        self._width = 14
        self._height = 14
        self._prob = {"0": 0.7, "1":0.3}

        self._border_size = 1
        self._border_tile = 1
        self._tile_size = 16
        self._graphics = None

        self._target_path = 50

        self._rewards = {
            "regions": 5,
            "path-length": 1
        }

    def adjust_param(self, **kwargs):
        self._width, self._height = kwargs.get('width', self._width), kwargs.get('height', self._height)
        self._prob["0"] = kwargs.get('empty_prob', self._prob["0"])
        self._prob["1"] = kwargs.get('solid_prob', self._prob["1"])
        kwargs["prob"] = self._prob

        self._target_path = kwargs.get('target_path', self._target_path)

        self._rewards = {
            "regions": kwargs.get('reward_regions', self._rewards["regions"]),
            "path-length": kwargs.get('reward_path_length', self._rewards["path-length"])
        }

    def get_stats(self, map):
        return {
            "regions": calc_num_regions(map, [0]),
            "path-length": calc_longest_path(map, [0])
        }

    def get_reward(self, new_stats, old_stats):
        #longer path is rewarded and less number of regions is rewarded
        rewards = {
            "regions": old_stats["regions"] - new_stats["regions"],
            "path-length": new_stats["path-length"] - old_stats["path-length"]
        }
        #unless the number of regions become zero, it has to be punished
        if new_stats["regions"] == 0 and old_stats["regions"] > 0:
            rewards["regions"] = -1
        #calculate the total reward
        return rewards["regions"] * self._rewards["regions"] +\
            rewards["path-length"] * self._rewards["path-length"]

    def get_episode_over(self, new_stats, old_stats):
        return new_stats["regions"] == 1 and new_stats["path-length"] >= self._target_path

    def get_debug_info(self, new_stats, old_stats):
        return {
            "regions": new_stats["regions"],
            "path-length": new_stats["path-length"]
        }

    def render(self, map):
        if self._graphics == None:
            self._graphics = {
                "0": Image.open(os.path.dirname(__file__) + "/binary/empty.png").convert('RGBA'),
                "1": Image.open(os.path.dirname(__file__) + "/binary/solid.png").convert('RGBA')
            }
        return super().render(map)
