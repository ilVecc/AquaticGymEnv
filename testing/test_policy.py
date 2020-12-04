import gym
import gym_aqua

from policies.dqn_policy import DQNPolicy
from policies.random_policy import RandomPolicy


def zero_to_infinity():
    time = 0
    while True:
        time += 1
        yield time


if __name__ == '__main__':
    
    # we already registered the environments in gym_aqua/__init__.py so now we're ready to go
    
    improve = True
    
    with gym.make('AquaSmall-v0') as env:
        
        policy = DQNPolicy("test_new_3/", env)
        # policy = RandomPolicy()
        if not policy.is_trained() or improve:
            policy.train()
        
        state = env.reset()
        episode_total_reward = 0
        env.render()
        for i in zero_to_infinity():
            # decide action
            pred_action = policy.get_action(state)
            # apply action
            state, reward, done, info = env.step(pred_action)
            episode_total_reward += reward
            env.render()
            
            if done:
                term_string = DQNPolicy.termination_string(info, episode_total_reward, i)
                print("{}".format(term_string))
                break
