import gym
import gym_aqua

import numpy as np


def policy(state):
    # best_action = env.action_space.sample()
    best_action = np.array([0.3, 0.3])  # + np.array(np.random.randint(-5, 6, 2) / 200)
    return best_action


def zero_to_infinity():
    time = 0
    while True:
        time += 1
        yield time


if __name__ == '__main__':
    # we register the environments in gym_aqua/__init__.py
    # so now we're ready
    
    env = gym.make('AquaSmall-v1')
    env.reset()  # not useful at all

    # TESTING
    state = env.reset()
    episode_total_reward = 0
    env.render()
    for i in zero_to_infinity():
        # decide action
        best_action = policy(state)
        # apply action
        state, reward, done, info = env.step(best_action)
        episode_total_reward += reward
        env.render()
    
        if done:
            # identify ending-state
            if info['Termination.stuck']:
                print("idling for too long [REW: {}]".format(episode_total_reward))
            elif info['Termination.collided']:
                print("collided ({} steps) [REW: {}]".format(i, episode_total_reward))
            elif info['Termination.time']:
                print("reached max steps [REW: {}]".format(episode_total_reward))
            else:
                print("---> REACHED GOAL ({} steps) [REW: {}]".format(i, episode_total_reward))
            break

    print(episode_total_reward)
