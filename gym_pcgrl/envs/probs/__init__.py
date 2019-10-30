from gym_pcgrl.envs.probs.binary_prob import BinaryProblem
from gym_pcgrl.envs.probs.ddave_prob import DDaveProblem
from gym_pcgrl.envs.probs.mdungeon_prob import MDungeonProblem
from gym_pcgrl.envs.probs.sokoban_prob import SokobanProblem

PROBLEMS = {
    "binary": BinaryProblem,
    "ddave": DDaveProblem,
    "mdungeon": MDungeonProblem,
    "sokoban": SokobanProblem
}
