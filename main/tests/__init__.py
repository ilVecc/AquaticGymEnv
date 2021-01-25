# noinspection PyUnresolvedReferences
import gym_aqua


class Policy(object):
    
    def get_action(self, state):
        raise NotImplementedError("no action selection strategy has been implemented")
    
    @staticmethod
    def zero_to_infinity():
        time = 0
        while True:
            time += 1
            yield time
    
    def test(self, env, render=True):
        step = 0
        episode_total_reward = 0
        info = {}
        
        state = env.reset()
        if render:
            env.render()
        for step in self.zero_to_infinity():
            # choose action
            pred_action = self.get_action(state)
            # apply action
            state, reward, done, info = env.step(pred_action)
            episode_total_reward += reward
            if render:
                env.render()
            if done:
                break
        
        return episode_total_reward, info, step
