from dqn import Normalizer


def aqua_term_string(info, episode_total_reward, steps):
    # identify ending-state
    if info['Termination.time']:
        msg = "reached max steps                "
    elif info['Termination.collided']:
        msg = "collided ({:3d} steps)             ".format(steps)
    else:
        msg = ">>> REACHED GOAL ({: 3d} steps) <<<".format(steps)
    return "{}  [REW: {: 7.2f}]".format(msg, episode_total_reward)


class AquaStateNormalizer(Normalizer):
    
    def __init__(self, env):
        self.env = env
        high_obs = self.env.observation_space.high
        low_obs = self.env.observation_space.low
        self.observation_range = high_obs - low_obs
    
    def __call__(self, state):
        norm_state = state.copy()
        norm_state /= self.observation_range  # set coordinates in [0,1] and angle in [-0.5,0.5]
        norm_state[2] += 0.5  # set angle in [0,1]
        return norm_state[None, :]  # None === np.newaxis
    
    def inverse(self, norm_state):
        state = norm_state.copy()
        state[0, 2] -= 0.5
        state *= self.observation_range
        return state[0, :]
