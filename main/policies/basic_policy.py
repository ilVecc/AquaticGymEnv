import gym


class AquaPolicy(object):
    
    def __init__(self, env_size, is_discrete, with_obstacles, with_waves):
        kwargs = {
            'with_obstacles': with_obstacles,
            'with_waves': with_waves
        }
        if env_size == 'huge':
            env_size = 'v2'
        elif env_size == 'medium':
            env_size = 'v1'
        else:
            env_size = 'v0'
            
        if is_discrete:
            self.env = gym.make("AquaEnv-{}".format(env_size), **kwargs)
        else:
            self.env = gym.make("AquaContinuousEnv-{}".format(env_size), **kwargs)

    def train(self, debug, render):
        pass

    def get_action(self, state):
        pass

    @staticmethod
    def zero_to_infinity():
        time = 0
        while True:
            time += 1
            yield time

    @staticmethod
    def termination_string(info, episode_total_reward, steps):
        # identify ending-state
        if info['Termination.time']:
            msg = "reached max steps                "
        elif info['Termination.collided']:
            msg = "collided ({:3d} steps)             ".format(steps)
        else:
            msg = " >>> REACHED GOAL ({:3d} steps) <<<".format(steps)
        return "{}  [REW: {:5.2f}]".format(msg, episode_total_reward)

    def test(self):
        state = self.env.reset()
        episode_total_reward = 0
        self.env.render()
        for i in AquaPolicy.zero_to_infinity():
            # choose action
            pred_action = self.get_action(state)
            # apply action
            state, reward, done, info = self.env.step(pred_action)
            episode_total_reward += reward
            self.env.render()
        
            if done:
                term_string = AquaPolicy.termination_string(info, episode_total_reward, i)
                print("{}".format(term_string))
                break
