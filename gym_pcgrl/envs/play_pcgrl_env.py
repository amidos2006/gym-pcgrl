from gym.spaces import Discrete
from gym_pcgrl.envs.pcgrl_env import PcgrlEnv

class PlayPcgrlEnv(PcgrlEnv):
    '''A designable and playable level.
    First, the designer creates a level. If it is playable, the player takes over.
    '''
    def __init__(self, prob='zeldaplay', rep='wide'):
        super().__init__(prob=prob, rep=rep)
        # 0 for designer, 1 for player
        self.active_agent = 0
        self.player_actions = [(1, 0), (0, 1), (-1, 0), (0, 0), (0, -1)]
        self.player_action_space = Discrete(len(self.player_actions))

    def step(self, action):
        if self.active_agent == 0:
            return super().step(action)
        print(action)
        move = self.player_actions[action]
        return self.play(move)

    def reset(self):
        if self.active_agent == 0: #and self._prob.playable:
            self.active_agent = 1
        else:
            self.active_agent = 0
        self._prob.active_agent = self.active_agent
        return super().reset()


    def play(self, move):
        print(self._rep._map)
        x, y = self._prob.player_coords
        trg = x, y + move
        self._rep.update(x, y, 0)
        self._prob.player_coords += move
        x, y = self._prob.player_coords
        self._rep.update(x, y, 'player')

