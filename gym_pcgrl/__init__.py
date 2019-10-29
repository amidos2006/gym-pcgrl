from gym.envs.registration import register

for rep in ["narrow", "wide", "turtle"]:
    register(
        id='binary-{}-v0'.format(rep),
        entry_point='gym_pcgrl.envs:BinaryEnv',
        kwargs={"rep": rep}
    )
    register(
        id='mdungeon-{}-v0'.format(rep),
        entry_point='gym_pcgrl.envs:MDungeonEnv',
        kwargs={"rep": rep}
    )
    register(
        id='ddave-{}-v0'.format(rep),
        entry_point='gym_pcgrl.envs:DDaveEnv',
        kwargs={"rep": rep}
    )
    register(
        id='sokoban-{}-v0'.format(rep),
        entry_point='gym_pcgrl.envs:SokobanEnv',
        kwargs={"rep": rep}
    )
