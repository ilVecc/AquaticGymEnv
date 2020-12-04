import os
import random
from collections import deque

import numpy as np
from keras import Sequential
from keras.engine.saving import model_from_json
from keras.layers import Dense
from keras.optimizers import Adam
from matplotlib import pyplot as plt

from policies.basic_policy import Policy


class NetworkUtils:
    
    @staticmethod
    def create_model(input_size, hidden_layer_size, output_size):
        """
        Create a dense neural network model with the given parameters

        Args:
            input_size: the number of nodes for the input layer
            output_size: the number of nodes for the output layer
            hidden_layer_size: the number of nodes for each hidden layer

        Returns:
            model: the corresponding neural network
        """
        model = Sequential()
        model.add(Dense(hidden_layer_size, input_dim=input_size, activation="relu"))
        model.add(Dense(hidden_layer_size, activation="relu"))
        model.add(Dense(output_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.0001))
        return model
    
    @staticmethod
    def save_network(network, folder):
        with open(os.path.join(folder, "model.json"), "w") as json_file:
            json_file.write(network.to_json())
        network.save_weights(os.path.join(folder, "model.h5"))
        print("Saved model to disk")
    
    @staticmethod
    def load_network(folder):
        with open(os.path.join(folder, "model.json"), 'r') as json_file:
            loaded_model = model_from_json(json_file.read())
        loaded_model.load_weights(os.path.join(folder, "model.h5"))
        loaded_model.compile(loss="mse", optimizer=Adam(learning_rate=0.0001))
        print("Loaded model from disk")
        return loaded_model
    
    @staticmethod
    def rolling(array, window):
        shape = array.shape[:-1] + (array.shape[-1] - window, window)
        strides = array.strides + (array.strides[-1],)
        return np.mean(np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides), -1)
    
    @staticmethod
    def show_network_performance(scores, folder=None):
        scores = np.array(scores)
        offset = 200
        episodes_offset = len(scores) - offset
        scores = scores[-offset:]
        
        plt.figure(figsize=(14, 8))
        
        episodes = np.arange(1, len(scores) + 1) + episodes_offset
        plt.plot(episodes, scores, label="score")
        
        scores_mean = np.full(scores.shape, np.mean(scores))
        episodes_mean = np.arange(1, len(scores_mean) + 1) + episodes_offset
        mean_plot = plt.plot(episodes_mean, scores_mean, label="mean")
        
        window_short = min(10, int(len(scores) * 0.25))
        scores_short = NetworkUtils.rolling(scores, window=window_short)
        episodes_short = episodes_mean[window_short:]
        plt.plot(episodes_short, scores_short, label="smooth")
        
        window_long = min(10, int(len(scores_short) * 0.25))
        scores_long = NetworkUtils.rolling(scores_short, window=window_long)
        episodes_long = episodes_short[window_long:]
        plt.plot(episodes_long, scores_long, label="very smooth")
        
        plt.yticks(list(plt.yticks()[0]) + [scores_mean[0]])
        plt.gca().get_yticklabels()[-1].set_color(mean_plot[0].get_color())
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.title("Reward per episode (last 200 episodes)")
        plt.legend()
        if folder is not None:
            plt.savefig(os.path.join(folder, "../test_new_3/reward_episode_recent.png"))
        plt.show()
    
    @staticmethod
    def show_overall_performance(scores, folder=None):
        scores = np.array(scores)
        episodes = np.arange(1, len(scores) + 1)
        
        plt.figure(figsize=(14, 8))
        
        factor = 0.1
        window_long = int(len(scores) * factor)
        scores_long = NetworkUtils.rolling(scores, window=window_long)
        episodes_long = episodes[window_long:]
        plt.plot(episodes_long, scores_long,
                 label="rolling window = {:2.0f}% episodes".format(factor * 100))
        
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.title("Overall Reward per episode")
        plt.legend()
        if folder is not None:
            plt.savefig(os.path.join(folder, "../test_new_3/reward_episode_overall.png"))
        plt.show()


class ReplayBuffer:
    """
    Constructs a buffer object that stores the past moves and samples a set of subsamples
    """
    
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
    
    def add(self, state, action, next_state, reward, done):
        """
        Add an experience to the buffer.

        Args:
            state: current state
            action: performed action
            next_state: next state
            reward: obtained reward
            done: whether it is the end of the episode
        """
        experience = (state[0], action, next_state[0], reward, done)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
    
    def size(self):
        return self.count
    
    def sample(self, batch_size):
        """Samples a total of elements equal to batch_size from buffer
        if buffer contains enough elements. Otherwise return all elements"""
        
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)
        
        # Maps each experience in batch to tuple (state, next_state, action, reward, done)
        return tuple(map(np.array, list(zip(*batch))))
    
    def clear(self):
        self.buffer.clear()
        self.count = 0


class AquaNormalizer(object):
    
    def __init__(self, env):
        self.env = env
        high_obs = np.concatenate((self.env.observation_space.high,
                                   self.env.observation_space.high[0:2]), axis=0)
        low_obs = np.concatenate((self.env.observation_space.low,
                                  self.env.observation_space.low[0:2]), axis=0)
        self.observation_range = high_obs - low_obs
        
    def __call__(self, state):
        state = np.concatenate((state, self.env.goal_state), axis=0)
        state /= self.observation_range  # set coordinates in [0,1] and angle in [-0.5,0.5]
        state[2] += 0.5  # set angle in [0,1]
        return state[np.newaxis, :]


class DQNImpl:
    
    def __init__(self, environment, discrete_actions, q_function, savedir):
        self.env = environment
        self.actions = discrete_actions
        self.q_function = q_function
        self.exp_buffer = ReplayBuffer(100000)
        self.checkpointdir = os.path.join(savedir, "checkpoint/")
        if not os.path.exists(self.checkpointdir):
            os.mkdir(self.checkpointdir)
        self.state_normalizer = AquaNormalizer(self.env)
        
    def _epsilon_greedy_best_action(self, state, epsilon):
        """
        Epsilon-greedy action selection function

        Args:
            state: agent's current state
            epsilon: epsilon parameter

        Returns:
            action id
        """
        n_actions = len(self.actions)
        q_actions = self.q_function.predict(state)
        pred_action = np.argmax(q_actions)
        probs = np.full(n_actions, epsilon / n_actions)
        probs[pred_action] += 1 - epsilon
        pred_action = np.random.choice(n_actions, p=probs)
        return pred_action, q_actions[0, pred_action]
    
    def _improve(self, gamma=0.95):
        batch_size = 256
        prediction_batch_size = int(batch_size / 4)
        b_state, b_action, b_next_state, b_reward, b_done = self.exp_buffer.sample(batch_size)
        
        # probability vector of the actions
        targets = self.q_function.predict(b_state[:], batch_size=prediction_batch_size)
        targets[np.arange(targets.shape[0]), b_action[:]] = b_reward
        # must consider next state if not terminal
        b_not_done = np.logical_not(b_done)
        future_quality = self.q_function.predict(b_next_state[b_not_done],
                                                 batch_size=prediction_batch_size)
        targets[b_not_done, b_action[b_not_done]] += gamma * np.max(future_quality, axis=1)
        
        # back-propagation
        return self.q_function.train_on_batch(b_state, targets)
    
    def train(self, trials, threshold=90, epsilon_decay=0.999995, debug=False, render=False):
        """
        Performs the Q-Learning algorithm for a specific environment on a specific neural network model

        Args:
            threshold: the desired overall reward for an episode
            trials: the number of iterations for the training phase
            epsilon_decay: the decay value of epsilon for the eps-greedy exploration
            debug: whether to print debug info
            render: whether to render the environment at each step

        Returns:
            score_queue: 1-d dimensional array of the reward obtained at each trial step
        """
        epsilon = 1.0
        epsilon_min = 0.01
        score_queue = []
        
        log = open(os.path.join(self.checkpointdir, "scores.txt"), "a")
        
        for epoch in range(trials):
            # reset environment
            state = self.state_normalizer(self.env.reset())
            episode_total_reward = 0
            steps = 0
            
            # simulate and improve
            done = False
            info = {}
            while not done:
                # epsilon greedy implementation (improves exploration)
                pred_action, pred_q_value = self._epsilon_greedy_best_action(state, epsilon)
                epsilon = max(epsilon * epsilon_decay, epsilon_min)  # stable on 0.01 at trial 92101
                
                # STEP
                # go to next state with selected action
                next_state, reward, done, info = self.env.step(self.actions[pred_action])
                next_state = self.state_normalizer(next_state)
                # update reward here
                episode_total_reward += reward
                
                if render:
                    self.env.render()
                
                # TRAIN MODEL
                # update experience buffer with what I've learned
                self.exp_buffer.add(state, pred_action, next_state, reward, done)
                # improve the Q function
                loss = self._improve()
                
                # if steps % 250 == 0:
                #     print("--> pred q-value: ", pred_q_value)
                #     print("    last loss:    ", loss)
                
                # move on to the next step
                state = next_state
                steps += 1
            
            # episode ended
            log.write(str(episode_total_reward) + "\n")
            log.flush()
            score_queue.append(episode_total_reward)
            
            # checkpoint
            if epoch % 100 == 0:
                print("\n------------------")
                NetworkUtils.save_network(self.q_function, self.checkpointdir)
                print("------------------\n")
            
            if debug:
                term_string = Policy.termination_string(info, episode_total_reward, steps)
                print("trial #{}: {}\n".format(epoch, term_string))
            
            # end training
            if np.mean(score_queue[-20:]) > threshold:
                # if goal is reached (with a large margin)
                print("Training concluded")
                break
        
        log.close()
        
        return self.q_function, score_queue


class DQNPolicy(Policy):
    
    def __init__(self, folder, env, thrusters_steps=2):
        super().__init__()
        self.env = env
        self.env.reset()
        self.state_normalizer = AquaNormalizer(self.env)

        # action space is continuous, but we can't handle it -> use discrete tuple
        self.actions = self._discretize_action_space(thrusters_steps)
        
        # check folder
        self.folder = folder
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)
        
        # get the Q function (modeled as a neural network)
        if len(os.listdir(self.folder)) != 0:
            # ALREADY BUILT AND TRAINED
            self.neural_policy = NetworkUtils.load_network(self.folder)
            self.trained = True
        else:
            # NON EXISTING AND TO BE TRAINED
            # 5 input [x_boat, y_boat, angle_boat, x_goal, y_goal]
            # 4 output (one per discretized action)
            input_size = self.env.observation_space.shape[0] + self.env.goal_state.shape[0]
            output_size = len(self.actions)
            self.neural_policy = NetworkUtils.create_model(input_size, 64, output_size)
            self.trained = False
    
    def _discretize_action_space(self, resolution):
        action_space_discrete = np.linspace(self.env.action_space.low,
                                            self.env.action_space.high,
                                            num=resolution, endpoint=True)
        return [(aL, aR)
                for aL in action_space_discrete[:, 0]
                for aR in action_space_discrete[:, 1]]
    
    def is_trained(self):
        return self.trained
    
    def train(self):
        learner = DQNImpl(self.env, self.actions, self.neural_policy, self.folder)
        neural_policy, episode_scores = learner.train(
            trials=1000000, threshold=90, debug=True, render=False)  # less than 100 steps
        
        NetworkUtils.save_network(neural_policy, self.folder)
        NetworkUtils.show_network_performance(episode_scores, self.folder)
        NetworkUtils.show_overall_performance(episode_scores, self.folder)
    
    def get_action(self, state):
        state = self.state_normalizer(state)
        actions_prob = self.neural_policy.predict(state)
        pred_action = self.actions[actions_prob.argmax()]
        return pred_action
