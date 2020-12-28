import os
import random
from collections import deque
from json import dump, load

import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from policies.basic_policy import AquaPolicy
from utils import assure_exists

# hyper-parameters and constants
TOT_EPISODES = 2000000
BUFFER_SIZE  =  100000  # full refreshes 20 times

EPSILON_FINAL = 0.01
EPSILON_INIT  = 1.0
EPSILON_DECAY = 500000  # overall observations after which the exploration is basically quit

MINIBATCH_SIZE  = 64
MIN_OBSERVATION = 6000

# threshold => reward 90 (< 100 steps) in last 20 episodes
THRESHOLD = 90

# network constants
INPUT_SIZE  = 5
OUTPUT_SIZE = 4


class DQNImpl:
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
    
    class Logger(object):
        
        def __init__(self, directory):
            self.filepath = os.path.join(directory, "scores.txt")
            self.file = None
        
        def start(self):
            self.file = open(self.filepath, "a")
        
        def log(self, value):
            self.file.write(str(value) + "\n")
            self.file.flush()
        
        def stop(self):
            self.file.close()
    
    class ReplayBuffer:
        """
        Constructs a buffer object that stores the past moves and samples a set of subsamples
        """
        
        def __init__(self, buffer_size):
            self._buffer = deque()
            self.buffer_size = buffer_size
            self.count = 0
        
        def add(self, s, a, r, sn, d):
            """
            Add an experience to the buffer.

            Args:
                s: current state
                a: performed action
                sn: next state
                r: obtained reward
                d: whether it is the end of the episode
            """
            experience = (s[0].tolist(), a, r, sn[0].tolist(), d)
            if self.count < self.buffer_size:
                self._buffer.append(experience)
                self.count += 1
            else:
                self._buffer.popleft()
                self._buffer.append(experience)
        
        def size(self):
            return self.count
        
        def sample_batch(self, batch_size):
            batch = random.sample(self._buffer, min(self.count, batch_size))
            # maps the tuples (s, a, sn, r, d) in the buffer
            # to a tuple of ndarrays
            return tuple(map(np.array, list(zip(*batch))))
        
        def clear(self):
            self._buffer.clear()
            self.count = 0

        def dump_experiences(self):
            return list(self._buffer)

        def load_experiences(self, experiences):
            self._buffer = deque(experiences)
            self.count = len(self._buffer)

    def __init__(self, environment, save_directory):
        self.env = environment
        self.q_function = self.create_model()
        self.exp_buffer = DQNImpl.ReplayBuffer(BUFFER_SIZE)
        self.state_normalizer = DQNImpl.StateNormalizer(self.env)
        
        # check folder
        self.working_directory = save_directory
        assure_exists(self.working_directory)
        self.checkpoint_directory = os.path.join(save_directory, "checkpoint/")
        assure_exists(self.checkpoint_directory)
        
        self.logger = DQNImpl.Logger(self.checkpoint_directory)
    
    def _epsilon_greedy_best_action(self, state, epsilon):
        """
        Epsilon-greedy action selection function

        Args:
            state: agent's current state
            epsilon: epsilon parameter

        Returns:
            action id
        """
        q_actions = self.q_function.predict(state)
        pred_action = np.argmax(q_actions)
        probs = np.full(OUTPUT_SIZE, epsilon / OUTPUT_SIZE)
        probs[pred_action] += 1 - epsilon
        pred_action = np.random.choice(OUTPUT_SIZE, p=probs)
        return pred_action, q_actions[0, pred_action]
    
    def _improve(self, batch_size=32, discount=0.99):
        """Update targets using batch prediction/training for each minibatch, so each subsequent
        minibatches predictions are made upon the updates of the previous minibatches"""
        
        b_state, b_action, b_reward, b_next_state, b_done = self.exp_buffer.sample_batch(batch_size)
        
        # probability vector of the actions
        targets = self.q_function.predict(b_state)
        future_quality = self.q_function.predict(b_next_state)
        # set required reward and consider next state if not terminal
        discount_selector = discount * np.logical_not(b_done)
        targets[np.arange(batch_size), b_action] = \
            b_reward + discount_selector * np.max(future_quality, axis=1)
        
        # back-propagation
        return self.q_function.train_on_batch(b_state, targets)
    
    def train(self, verbose=False, render=False):
        """
        Performs Q-Learning algorithm for the given environment on this neural network model

        Args:
            verbose: whether to print debug info
            render: whether to render the environment at each step

        Returns:
            score_queue: 1-d dimensional array of the overall reward obtained in each episode
        """
        epsilon_pace = (EPSILON_INIT - EPSILON_FINAL) / EPSILON_DECAY
        epsilon = EPSILON_INIT
        score_queue = []
        
        self.logger.start()
        
        for episode in range(1, TOT_EPISODES + 1):
            
            # reset environment
            state = self.state_normalizer(self.env.reset())
            episode_total_reward = 0
            episode_steps = 0
            done = False
            info = {}
            
            #
            # SIMULATE AND IMPROVE
            #
            while not done:
                # CHOOSE STATE
                # epsilon greedy implementation (improves exploration)
                pred_action, pred_q_value = self._epsilon_greedy_best_action(state, epsilon)
                epsilon = max(epsilon - epsilon_pace, EPSILON_FINAL)
                
                # STEP
                # go to next state with selected action
                next_state, reward, done, info = self.env.step(pred_action)
                next_state = self.state_normalizer(next_state)
                episode_total_reward += reward
                
                if render:
                    self.env.render()
                
                # TRAIN MODEL
                # update experience buffer with what I've learned and improve the Q function
                self.exp_buffer.add(state, pred_action, reward, next_state, done)
                if self.exp_buffer.size() > MIN_OBSERVATION:
                    self._improve(MINIBATCH_SIZE)
                
                # move on to the next step
                state = next_state
                episode_steps += 1
            
            # episode ended
            self.logger.log(episode_total_reward)
            score_queue.append(episode_total_reward)
            
            #
            # EPISODE RESULTS
            #
            
            # checkpoint
            if episode % 100 == 0:
                if verbose:
                    # noinspection PyUnboundLocalVariable
                    print("\n--> pred q-value: ", pred_q_value)
                    print("    last epsilon: ", epsilon)
                    print()
                print("\n------------------")
                self.save_network()
                self.save_buffer()
                print("------------------\n\n")
            
            if verbose:
                term_string = AquaPolicy.termination_string(
                    info, episode_total_reward, episode_steps)
                state = self.state_normalizer.inverse(state)
                print("episode #{}: {}  (x:{:3.1f} y:{:3.1f} a:{:4.1f}°)\n".format(
                    episode, term_string, state[0, 0], state[0, 1], state[0, 2] / np.pi * 180))
            
            # end training
            if np.mean(score_queue[-20:]) > THRESHOLD:
                # if goal is reached (with a large margin)
                print("Training concluded")
                break
        
        self.logger.stop()
        self.save_network()
        self.save_buffer()
        
        return score_queue
    
    def predict(self, state):
        state = self.state_normalizer(state)
        actions_prob = self.q_function.predict(state)
        pred_action = actions_prob.argmax()
        return pred_action
    
    @staticmethod
    def create_model():
        """
        Create a dense neural network model for the discrete Aqua environment

        Returns:
            model: the corresponding neural network
        """
        model = Sequential()
        model.add(Dense(input_dim=INPUT_SIZE,
                        units=64,
                        activation="relu"))
        model.add(Dense(units=64,
                        activation="relu"))
        model.add(Dense(units=OUTPUT_SIZE,
                        activation="linear"))
        model.compile(loss="mse",
                      optimizer=Adam(learning_rate=0.001))
        return model
    
    def save_network(self):
        self.q_function.save_weights(os.path.join(self.checkpoint_directory, "model.h5"))
        print("Saved model to disk")
    
    def save_buffer(self):
        with open(os.path.join(self.checkpoint_directory, 'exp_buff.json'), 'w') as exp_buff_file:
            dump(self.exp_buffer.dump_experiences(), exp_buff_file, indent='\t')
    
    def load_network(self):
        self.q_function.load_weights(os.path.join(self.checkpoint_directory, "model.h5"))
        print("Loaded model from disk")
    
    def load_buffer(self):
        with open(os.path.join(self.checkpoint_directory, 'exp_buff.json'), 'r') as exp_buff_file:
            self.exp_buffer.load_experiences(load(exp_buff_file))


class DQNAquaPolicy(AquaPolicy):
    
    def __init__(self, folder, load_network, env_size, with_obstacles, with_waves):
        super().__init__(env_size, True, with_obstacles, with_waves)
        
        # build the Q function (modeled as a neural network)
        self.dqn = DQNImpl(self.env, folder)
        if load_network:
            self.dqn.load_network()
            self.dqn.load_buffer()
    
    def train(self, debug, render):
        return self.dqn.train(verbose=debug, render=render)
    
    def get_action(self, state):
        return self.dqn.predict(state)
