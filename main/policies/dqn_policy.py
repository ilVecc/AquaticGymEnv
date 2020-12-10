import os
import random
from collections import deque

import gym
import numpy as np
from keras import Sequential
from keras.engine.saving import model_from_json
from keras.layers import Dense
from keras.optimizers import Adam

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


class StateNormalizer(object):
    
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
    
    def inverse(self, state):
        state = state[0, 0:3]
        state[2] -= 0.5
        state *= self.observation_range[0:3]
        return state[np.newaxis, :]


class DQNImpl:
    
    def __init__(self, environment, q_function, savedir):
        self.env = environment
        self.q_function = q_function
        self.exp_buffer = ReplayBuffer(100000)
        self.checkpointdir = os.path.join(savedir, "checkpoint/")
        if not os.path.exists(self.checkpointdir):
            os.mkdir(self.checkpointdir)
        self.state_normalizer = StateNormalizer(self.env)
    
    def _epsilon_greedy_best_action(self, state, epsilon):
        """
        Epsilon-greedy action selection function

        Args:
            state: agent's current state
            epsilon: epsilon parameter

        Returns:
            action id
        """
        n_actions = self.env.action_space.n
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
        Performs Q-Learning algorithm for the given environment on this neural network model

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
                next_state, reward, done, info = self.env.step(pred_action)
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
                state = self.state_normalizer.inverse(state)
                print("trial #{}: {}  (x:{:2.1f} y:{:2.1f} a:{:3.1f}°)\n".format(
                    epoch, term_string, state[0, 0], state[0, 1], state[0, 2] / np.pi * 180))
            
            # end training
            if np.mean(score_queue[-20:]) > threshold:
                # if goal is reached (with a large margin)
                print("Training concluded")
                break
        
        log.close()
        
        return self.q_function, score_queue


class DQNPolicy(Policy):
    
    def __init__(self, env, folder):
        super().__init__(env)
        
        assert isinstance(self.env.action_space, gym.spaces.Discrete), \
            "DQN cannot be used on non-discrete action spaces!"
        
        self.env.reset()
        self.state_normalizer = StateNormalizer(self.env)
        
        # check folder
        self.folder = folder
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)
        
        # get the Q function (modeled as a neural network)
        if 'model.h5' in os.listdir(self.folder):
            # ALREADY BUILT AND TRAINED
            self.neural_policy = NetworkUtils.load_network(self.folder)
            self.trained = True
        else:
            # NON EXISTING AND TO BE TRAINED
            # 5 input [x_boat, y_boat, angle_boat, x_goal, y_goal]
            # one output per discrete action
            input_size = self.env.observation_space.shape[0] + self.env.goal_state.shape[0]
            output_size = self.env.action_space.n
            self.neural_policy = NetworkUtils.create_model(input_size, 64, output_size)
            self.trained = False
    
    def is_trained(self):
        return self.trained
    
    def train(self):
        learner = DQNImpl(self.env, self.neural_policy, self.folder)
        # threshold => reward 90 (< 100 steps) in last 20 episodes
        neural_policy, episodes_score = learner.train(
            trials=1000000, threshold=90, debug=True, render=False)
        NetworkUtils.save_network(neural_policy, self.folder)
        return neural_policy, episodes_score
    
    def get_action(self, state):
        state = self.state_normalizer(state)
        actions_prob = self.neural_policy.predict(state)
        pred_action = actions_prob.argmax()
        return pred_action
