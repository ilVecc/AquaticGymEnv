from gym.envs.registration import register

register(
    id='AquaSmall-v1',
    entry_point='gym_aqua.envs:AquaSmallEnv',
    kwargs={'scenario': 0},
    # max_episode_steps=200,
    # reward_threshold=25.0
)

register(
    id='AquaMedium-v0',
    entry_point='gym_aqua.envs:AquaMediumEnv'
)

register(
    id='AquaHuge-v0',
    entry_point='gym_aqua.envs:AquaHugeEnv'
)
