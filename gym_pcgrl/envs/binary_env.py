from gym_pcgrl.envs.helper import calc_num_regions, calc_longest_path
import os
from PIL import Image
from gym_pcgrl.envs.pcgrl_env import PcgrlEnv

class BinaryEnv(PcgrlEnv):
    def _calc_rep_stats(self):
        self._rep_stats = {
            "regions": calc_num_regions(self._rep._map, [0]),
            "path-length": calc_longest_path(self._rep._map, [0])
        }

    def _init_param(self, **kwargs):
        self._rep._init_param(14, 14, {"0": 0.7, "1":0.3})

        self._target_path = 50

        self._rewards = {
            "regions": 5,
            "path-length": 1
        }

    def adjust_param(self, **kwargs):
        empty_prob = kwargs.get('empty_prob', self._rep._prob["0"])
        solid_prob = kwargs.get('solid_prob', self._rep._prob["1"])
        kwargs["prob"] = {"0":empty_prob, "1":solid_prob}
        kwargs["width"], kwargs["height"] = kwargs.get('width', self._rep._width), kwargs.get('height', self._rep._height)
        super().adjust_param(**kwargs)

        self._target_path = kwargs.get('target_path', self._target_path)

        self._rewards = {
            "regions": kwargs.get('reward_regions', self._rewards["regions"]),
            "path-length": kwargs.get('reward_path_length', self._rewards["path-length"])
        }

    def _calc_total_reward(self, old_stats):
        #longer path is rewarded and less number of regions is rewarded
        rewards = {
            "regions": old_stats["regions"] - self._rep_stats["regions"],
            "path-length": self._rep_stats["path-length"] - old_stats["path-length"]
        }
        #unless the number of regions become zero, it has to be punished
        if self._rep_stats["regions"] == 0 and old_stats["regions"] > 0:
            rewards["regions"] = -1
        #calculate the total reward
        return rewards["regions"] * self._rewards["regions"] +\
            rewards["path-length"] * self._rewards["path-length"]

    def _calc_episode_over(self, old_stats):
        return self._rep_stats["regions"] == 1 and self._rep_stats["path-length"] >= self._target_path

    def _calc_debug_info(self, old_stats):
        return {
            "regions": self._rep_stats["regions"],
            "path-length": self._rep_stats["path-length"]
        }

    def render(self, mode='human'):
        tile_size = 16
        graphics = {
            "0": Image.open(os.path.dirname(__file__) + "/binary/empty.png").convert('RGBA'),
            "1": Image.open(os.path.dirname(__file__) + "/binary/solid.png").convert('RGBA')
        }
        return super().render(graphics, 1, tile_size, mode)
