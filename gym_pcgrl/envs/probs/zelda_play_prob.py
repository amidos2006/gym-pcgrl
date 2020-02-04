from gym_pcgrl.envs.probs.zelda_prob import ZeldaProblem

class ZeldaPlayProblem(ZeldaProblem):
    ''' A version of zelda in which a player may control Link and play the game.'''
    def __init__(self):
        super().__init__()
        self._width = 16
        self._height = 16
