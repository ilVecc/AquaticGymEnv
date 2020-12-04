

class Policy(object):
    
    def __init__(self):
        pass

    def is_trained(self):
        pass

    def train(self):
        pass

    def get_action(self, state):
        pass

    @staticmethod
    def termination_string(info, episode_total_reward, steps):
        # identify ending-state
        if info['Termination.time']:
            return "reached max steps [REW: {}]".format(episode_total_reward)
        elif info['Termination.stuck']:
            return "idling for too long [REW: {}]".format(episode_total_reward)
        elif info['Termination.collided']:
            return "collided ({} steps) [REW: {}]".format(steps, episode_total_reward)
        else:
            return " ---> REACHED GOAL ({} steps) [REW: {}]".format(steps, episode_total_reward)
