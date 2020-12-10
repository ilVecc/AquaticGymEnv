

class Policy(object):
    
    def __init__(self, env):
        self.env = env

    def is_trained(self):
        pass

    def train(self):
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
        return "{}  [REW: {:.2f}]".format(msg, episode_total_reward)

    def test(self):
        state = self.env.reset()
        episode_total_reward = 0
        self.env.render()
        for i in Policy.zero_to_infinity():
            # choose action
            pred_action = self.get_action(state)
            # apply action
            state, reward, done, info = self.env.step(pred_action)
            episode_total_reward += reward
            self.env.render()
        
            if done:
                term_string = Policy.termination_string(info, episode_total_reward, i)
                print("{}".format(term_string))
                break
