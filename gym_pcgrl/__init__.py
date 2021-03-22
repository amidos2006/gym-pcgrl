from gym.envs.registration import register
from gym_pcgrl.envs.probs import PROBLEMS
from gym_pcgrl.envs.reps import REPRESENTATIONS

# Register all the problems with every different representation for the OpenAI GYM
for prob in PROBLEMS.keys():
    for rep in REPRESENTATIONS.keys():
        if 'play' in prob:
            entry_point='gym_pcgrl.envs:PlayPcgrlEnv'
        else:
            entry_point='gym_pcgrl.envs:PcgrlEnv'
        register(
            id='{}-{}-v0'.format(prob, rep),
            entry_point=entry_point,
            kwargs={"prob": prob, "rep": rep}
        )
