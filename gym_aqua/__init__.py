from gym.envs.registration import register

register(
    id='AquaEnv-v0',
    entry_point='gym_aqua.envs:AquaEnv'
)
register(
    id='AquaContinuousEnv-v0',
    entry_point='gym_aqua.envs:AquaContinuousEnv'
)
