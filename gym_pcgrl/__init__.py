from gym.envs.registration import register

for rep in ["narrow", "wide", "turtle"]:
    register(
        id='binary-{}-v0'.format(rep),
        entry_point='gym_pcgrl.envs:BinaryEnv',
        kwargs={"rep": rep}
    )
    register(
        id='dungeon-{}-v0'.format(rep),
        entry_point='gym_pcgrl.envs:DungeonEnv',
        kwargs={"rep": rep}
    )
    register(
        id='platformer-{}-v0'.format(rep),
        entry_point='gym_pcgrl.envs:PlatformerEnv',
        kwargs={"rep": rep}
    )
    register(
        id='sokoban-{}-v0'.format(rep),
        entry_point='gym_pcgrl.envs:SokobanEnv',
        kwargs={"rep": rep}
    )
