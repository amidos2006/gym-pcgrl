from gym_pcgrl.envs.probs.binary_prob import BinaryProblem, MultiGoalBinaryProblem
from gym_pcgrl.envs.probs.ddave_prob import DDaveProblem
from gym_pcgrl.envs.probs.mdungeon_prob import MDungeonProblem
from gym_pcgrl.envs.probs.sokoban_prob import SokobanProblem
from gym_pcgrl.envs.probs.zelda_prob import ZeldaProblem, MultiGoalZeldaProblem
from gym_pcgrl.envs.probs.smb_prob import SMBProblem
from gym_pcgrl.envs.probs.zelda_play_prob import ZeldaPlayProblem

# all the problems should be defined here with its corresponding class
PROBLEMS = {
    "binary": BinaryProblem,
    "binarygoal": MultiGoalBinaryProblem,
    "ddave": DDaveProblem,
    "mdungeon": MDungeonProblem,
    "sokoban": SokobanProblem,
    "zelda": ZeldaProblem,
    "zeldagoal": MultiGoalZeldaProblem,
    "smb": SMBProblem,
    "zeldaplay": ZeldaPlayProblem,
}
