from gym.envs.registration import register

# SMALL
register(
    id='AquaSmallEnv-v0',
    entry_point='gym_aqua.envs:AquaSmall',
    kwargs={'with_obstacles': False}
)
register(
    id='AquaSmallEnv-v1',
    entry_point='gym_aqua.envs:AquaSmall',
    kwargs={'with_obstacles': True}
)
register(
    id='AquaSmallContinuousEnv-v0',
    entry_point='gym_aqua.envs:AquaSmallContinuous',
    kwargs={'with_obstacles': False}
)
register(
    id='AquaSmallContinuousEnv-v1',
    entry_point='gym_aqua.envs:AquaSmallContinuous',
    kwargs={'with_obstacles': True}
)

# MEDIUM
register(
    id='AquaMediumEnv-v0',
    entry_point='gym_aqua.envs:AquaMedium',
    kwargs={'with_obstacles': False}
)
register(
    id='AquaMediumEnv-v1',
    entry_point='gym_aqua.envs:AquaMedium',
    kwargs={'with_obstacles': True}
)
register(
    id='AquaMediumContinuousEnv-v0',
    entry_point='gym_aqua.envs:AquaMediumContinuous',
    kwargs={'with_obstacles': False}
)
register(
    id='AquaMediumContinuousEnv-v1',
    entry_point='gym_aqua.envs:AquaMediumContinuous',
    kwargs={'with_obstacles': True}
)

# HUGE
register(
    id='AquaHugeEnv-v0',
    entry_point='gym_aqua.envs:AquaHuge',
    kwargs={'with_obstacles': False}
)
register(
    id='AquaHugeEnv-v1',
    entry_point='gym_aqua.envs:AquaHuge',
    kwargs={'with_obstacles': True}
)
register(
    id='AquaHugeContinuousEnv-v0',
    entry_point='gym_aqua.envs:AquaHugeContinuous',
    kwargs={'with_obstacles': False}
)
register(
    id='AquaHugeContinuousEnv-v1',
    entry_point='gym_aqua.envs:AquaHugeContinuous',
    kwargs={'with_obstacles': True}
)
