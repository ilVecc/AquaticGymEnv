import gym
import gym_aqua

from testing.dqn import *


def discretize_action_space(environment, resolution):
    action_space_discrete = np.linspace(environment.action_space.low,
                                        environment.action_space.high,
                                        num=resolution, endpoint=True)
    return [(aL, aR) for aL in action_space_discrete[:, 0] for aR in action_space_discrete[:, 1]]


def zero_to_infinity():
    time = 0
    while True:
        time += 1
        yield time


if __name__ == '__main__':
    
    # we already registered the environments in gym_aqua/__init__.py so now we're ready to go
    
    with gym.make('AquaSmall-v0') as env:
        env.reset()
        
        # action space is continuous, but we can't handle it -> use discrete tuple
        actions = discretize_action_space(env, resolution=6)
        
        folder = "test_improve_5_linear2/"
        if not os.path.exists(folder):
            os.mkdir(folder)

        # get the Q function (modeled as a neural network)
        if len(os.listdir(folder)) != 0:
            neural_policy = DQNUtils.load_network(folder)
            improve = False
        else:
            # 3 input [x_boat, y_boat, angle_boat]
            # 16 output (one per action)
            neural_policy = DQNUtils.create_model(env.observation_space.shape[0], 64, len(actions))
            improve = True
        
        if improve:
            neural_policy, episode_scores = \
                DQN(env, actions, neural_policy, folder) \
                    .train(trials=1000000, debug=True, render=False)
            
            DQNUtils.save_network(neural_policy, folder)
            DQNUtils.show_network_performance(episode_scores, folder)
        
        # TESTING
        state = env.reset()
        episode_total_reward = 0
        env.render()
        for i in zero_to_infinity():
            # decide action
            state = state[np.newaxis, :]
            actions_prob = neural_policy.predict(state)
            pred_action = actions[actions_prob.argmax()]
            # apply action
            state, reward, done, info = env.step(pred_action)
            episode_total_reward += reward
            env.render()
    
            if done:
                term_string = DQNUtils.termination_string(info, episode_total_reward, i)
                print("{}".format(term_string))
                break
