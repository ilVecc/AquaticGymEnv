import numpy as np
from gym.envs.registration import register

register(
    id='AquaEnv-v0',
    entry_point='gym_aqua.envs:AquaEnv'
)
register(
    id='AquaContinuousEnv-v0',
    entry_point='gym_aqua.envs:AquaContinuousEnv'
)

register(
    id='AquaEnv-v1',
    entry_point='gym_aqua.envs:AquaEnv',
    kwargs={'obstacles': True}
)
register(
    id='AquaContinuousEnv-v1',
    entry_point='gym_aqua.envs:AquaContinuousEnv',
    kwargs={'obstacles': True}
)

difficult_obstacles = [
    (np.array([15, 70]), "c", 5),
    (np.array([25, 40]), "r", (10, 10)),
    (np.array([40, 80]), "r", (10, 10)),
    (np.array([55, 20]), "c", 10),
    (np.array([60, 55]), "r", (20, 20)),
    (np.array([85, 75]), "c", 5)
]
register(
    id='AquaEnv-v2',
    entry_point='gym_aqua.envs:AquaEnv',
    kwargs={'obstacles': difficult_obstacles}
)
register(
    id='AquaContinuousEnv-v2',
    entry_point='gym_aqua.envs:AquaContinuousEnv',
    kwargs={'obstacles': difficult_obstacles}
)
