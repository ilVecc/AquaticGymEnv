from gym.envs.registration import register

# SMALL
register(
    id='AquaEnv-v0',
    entry_point='gym_aqua.envs:AquaSmall'
)
register(
    id='AquaContinuousEnv-v0',
    entry_point='gym_aqua.envs:AquaSmallContinuous'
)

# MEDIUM
register(
    id='AquaEnv-v1',
    entry_point='gym_aqua.envs:AquaMedium'
)
register(
    id='AquaContinuousEnv-v1',
    entry_point='gym_aqua.envs:AquaMediumContinuous'
)

# HUGE
register(
    id='AquaEnv-v2',
    entry_point='gym_aqua.envs:AquaHuge'
)
register(
    id='AquaContinuousEnv-v2',
    entry_point='gym_aqua.envs:AquaHugeContinuous'
)
