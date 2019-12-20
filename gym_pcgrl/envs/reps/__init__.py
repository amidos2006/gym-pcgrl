from gym_pcgrl.envs.reps.narrow_rep import NarrowRepresentation
# from gym_pcgrl.envs.reps.narrow_cast_rep import NarrowCastRepresentation
from gym_pcgrl.envs.reps.wide_rep import WideRepresentation
from gym_pcgrl.envs.reps.turtle_rep import TurtleRepresentation

# all the representations should be defined here with its corresponding class
REPRESENTATIONS = {
    "narrow": NarrowRepresentation,
    # "narrowcast": NarrowCastRepresentation,
    "wide": WideRepresentation,
    "turtle": TurtleRepresentation
}
