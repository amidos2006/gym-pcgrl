import numpy as np
from gym.spaces import Discrete

from gym_pcgrl.envs.helper import get_int_prob, get_string_map
from gym_pcgrl.envs.pcgrl_env import PcgrlEnv


class PlayPcgrlEnv(PcgrlEnv):
    '''A designable and playable level.
    First, the designer creates a level. If it is playable, the player takes over.
    '''
    def __init__(self, prob='zeldaplay', rep='wide'):
        super().__init__(prob=prob, rep=rep)
        # 0 for designer, 1 for player
        # whose turn it should be
        self.trg_agent = 0
        # whose turn it is
        self.active_agent = 0
        self.player_actions = [(1, 0),
                (0, 1), (-1, 0),
               #(0, 0),
                (0, -1)]
        self.player_action_space = Discrete(len(self.player_actions))
        self.player_rew = 0
        self.next_rep_map = None
        self.player_coords = None

    def get_player_action_space(self):
        return self.player_action_space

    def step(self, action):
       #print(self.active_agent)
        if self.active_agent == 0:
            obs, rew, done, info = super().step(action)
        elif self.active_agent == 1:
           #print('player coords: {}'.format(self._prob.player_coords))
            if self.next_rep_map is not None:
                self._rep._map = self.next_rep_map
                self._next_rep_map = None
                # for player coords
                self._rep_stats = self._prob.get_stats(get_string_map(self._rep._map, self._prob.get_tile_types()))
           #self.
            action = np.ravel_multi_index(action, (self.h, self.w, self.dim))
            move = self.player_actions[action]
            obs, rew, done, info = self.play(move)
        if self._prob.playable:
            info['trg_agent'] = self.trg_agent
            info['playable_map'] = self._rep._map
            # won't be overwritten by reset
            self._next_rep_map = self._rep._map

        return obs, rew, done, info

   #def set_map(self, map):
   #    self.next_rep_map = map
       #self._rep_stats = self._prob.get_stats(get_string_map(self._rep._map, self._prob.get_tile_types()))
       #self.
       #self._prob.reset(self._rep_stats)


    def reset(self):
        self.player_rew = 0
        if self.active_agent == 0 and self._prob.playable: #and self._prob.playable:
            self.trg_agent = 1
            self._rep_stats = self._prob.get_stats(get_string_map(self._rep._map, self._prob.get_tile_types()))
            self._heatmap = np.zeros((self._prob._height, self._prob._width))

            observation = self._rep.get_observation()
            observation["heatmap"] = self._heatmap.copy()

            return observation
        else:
            self.trg_agent = 0
            self._prob.trg_agent = 0

            return super().reset()

    def set_active_agent(self, n_agent):
        self.active_agent = n_agent
        self._prob.active_agent = n_agent

        return

    def play(self, move):
        if self._prob.player_coords is None:
            self._rep_stats = self._prob.get_stats(get_string_map(self._rep._map, self._prob.get_tile_types()))
            if self._prob.player_coords is None:
                self._prob.player_coords = 3, 3
        assert self._prob.player_coords is not None
        x, y = self._prob.player_coords
        tile_types = self._prob.get_tile_types()
        player_chan = self._rep._map[y, x]
       #assert tile_types[player_chan] == 'player'
        dx, dy = move[0], move[1]
        x_t, y_t = x + dx, y + dy
        if x_t >= self._prob._width or y_t >= self._prob._width or x_t < 0 or y_t < 0:
            # check for out of bounds
            pass
        else:
            trg_chan = self._rep._map[y_t, x_t]
            # impassable tiles

            if trg_chan == 1:
                pass
            elif trg_chan == 3:
                self._prob.play_rew += 1
            if trg_chan in [0, 3]:
                self._rep.update([x, y, 0])
                self._rep.update([x_t, y_t, 2])
                self._prob.player_coords = x_t, y_t
        obs = self._rep.get_observation()
        rew = self._prob.get_reward(None, None)
        done = False
        info = {}

        return obs, rew, done, info
