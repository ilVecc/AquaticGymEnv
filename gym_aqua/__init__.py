from gym.envs.registration import register


register(
    id='AquaSmall-v0',
    entry_point='gym_aqua.envs:AquaSmallEnv',
    kwargs={'with_obstacles': True}
)

register(
    id='AquaSmall-v1',
    entry_point='gym_aqua.envs:AquaSmallEnv',
    kwargs={'with_obstacles': False}
)

register(
    id='AquaMedium-v0',
    entry_point='gym_aqua.envs:AquaMediumEnv'
)

register(
    id='AquaHuge-v0',
    entry_point='gym_aqua.envs:AquaHugeEnv'
)
