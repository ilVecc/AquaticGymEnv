import os

from testing import Policy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import gym
import numpy as np
import tensorflow as tf

from impl.utils import aqua_term_string, AquaStateNormalizer


class DQNAquaPolicy(Policy):
    def __init__(self, savepath, normalizer):
        self.normalizer = (lambda x: x[np.newaxis, ...]) if not normalizer else normalizer
        # get the DQN trained network
        self.network = tf.keras.models.load_model(savepath)
    
    def get_action(self, state):
        state = self.normalizer(state)
        return self.network(state).numpy().argmax()


if __name__ == '__main__':
    # we already registered the environments in gym_aqua/__init__.py so now we're ready to go
    
    env = gym.make("AquaEnv-v0", obstacles=True)
    policy = DQNAquaPolicy(savepath='example_policies/example_no_obs/models/model-00030', normalizer=AquaStateNormalizer(env))
    tot_rew, info, tot_steps = policy.test(env)
    print(aqua_term_string(info, tot_rew, tot_steps))
