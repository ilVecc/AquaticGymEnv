import os
import random
from collections import deque
from json import dump, load
from time import perf_counter
from typing import Union

import numpy as np
import tensorflow as tf

from policies.basic_policy import AquaPolicy
from utils import assure_exists, get_time

# hyper-parameters and constants
TOT_EPISODES = 10000
BUFFER_SIZE = 50000

EPSILON_FINAL = 0.01
EPSILON_INIT = 1.0
EPSILON_DECAY = 0.9999  # either a factor or a number of episodes

MINIBATCH_SIZE = 64
MIN_OBSERVATION = 6000
GAMMA = 0.98
TAU = 0.005

WEIGHTS_PREFIX = "checkpoint"
EXP_BUFFER_PREFIX = "experience"

#
# LIBRARY DETAILS
#
# Tensorflow 2.x better handles prediction on small batches using __call__(); use predict() only
# when using big batches. Since here we always ask for either 1 or MINIBATCH_SIZE predictions,
# we will be using __call__()
#
# You can check by yourself that doing so saves a ton of time
#


class DQN:
    
    def __init__(self, environment, save_directory, normalizer=None):
        # environment related
        self.env = environment
        self.input_shape = self.env.observation_space.shape
        self.action_space = self.env.action_space.n
        # learning related
        self.q_function = self.create_model()
        self.q_function_target = self.create_model()
        self.q_function_target.set_weights(self.q_function.get_weights())
        self.exp_buffer = deque(maxlen=BUFFER_SIZE)
        self.state_normalizer = (lambda x: x) if not normalizer else normalizer
        
        # check folder
        self.working_directory = save_directory
        assure_exists(self.working_directory)
        self.checkpoint_directory = os.path.join(save_directory, "checkpoint/")
        assure_exists(self.checkpoint_directory)
        
        self.checkpoint_counter = 1
    
    def _epsilon_greedy_best_action(self, state, epsilon):
        """
        Epsilon-greedy action selection function

        Args:
            state: agent's current state
            epsilon: epsilon parameter

        Returns:
            action id, action quality
        """
        q_actions = self.q_function(state)
        pred_action = np.argmax(q_actions)
        probs = np.full(self.action_space, epsilon / self.action_space)
        probs[pred_action] += 1 - epsilon
        pred_action = np.random.choice(self.action_space, p=probs)
        return pred_action, q_actions[0, pred_action]
    
    def _improve_network(self, batch_size=32, gamma=0.99):
        """Update targets using batch prediction/training for each minibatch, so each subsequent
        minibatches predictions are made upon the updates of the previous minibatches"""
        batch_size = min(len(self.exp_buffer), batch_size)
        batch = list(zip(*random.sample(self.exp_buffer, batch_size)))
        # maps the tuples (s, a, r, sn, d) in the buffer to a tuple of np.ndarrays
        b_state = np.vstack(batch[0])
        b_action = np.array(batch[1])
        b_reward = np.array(batch[2])
        b_next_state = np.vstack(batch[3])
        b_done = np.array(batch[4])
        
        targets = self._improve_strategy_double(
            gamma, b_state, b_action, b_reward, b_next_state, b_done)
        
        # back-propagation
        return self.q_function.train_on_batch(b_state, targets)
    
    # def _other_diff(self, targets):
    #     prediction_value = self.q_function(b_state)[np.arange(batch_size), b_action]
    #     mse = tf.math.square(prediction_value - target_value)
    #     objective_function = tf.math.reduce_mean(mse)
    #
    #     with tf.GradientTape() as tape:
    #         # Compute loss with custom loss function
    #         objective_function = self.actor_objective_function_double(samples)
    #         # Compute gradients actor for network
    #         grads = tape.gradient(objective_function, self.actor.trainable_variables)
    #         # Apply gradients to update network weights
    #         self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
    
    def _improve_strategy_double(self, gamma, b_state, b_action, b_reward, b_next_state, b_done):
        batch_size = b_state.shape[0]
        # double strategy:     dE(w)/dw = (Qw(s,a) - (r + y * Qwt(s',a')) dQw(s,a)/dw
        #                                                      with  a' = argmax_A Qw(s',A)
        #                      w <- w - alpha * dE(w)/dw
        best_next_actions = np.argmax(self.q_function(b_state).numpy(), axis=1)
        future_quality = self.q_function_target(b_next_state).numpy()[np.arange(batch_size),
                                                               best_next_actions]
        # set required reward and consider next state if not terminal
        targets = self.q_function(b_state).numpy()
        targets[np.arange(batch_size), b_action] = \
            b_reward + np.logical_not(b_done) * gamma * future_quality
        return targets
    
    def _improve_strategy_fixed(self, gamma, b_state, b_action, b_reward, b_next_state, b_done):
        batch_size = b_state.shape[0]
        # fixed strategy:      dE(w)/dw = (Qw(s,a) - (r + y * max_A Qwt(s',A)) dQw(s,a)/dw
        #                      w <- w - alpha * dE(w)/dw
        future_quality = self.q_function_target(b_next_state)
        # set required reward and consider next state if not terminal
        targets = self.q_function(b_state)
        targets[np.arange(batch_size), b_action] = \
            b_reward + np.logical_not(b_done) * gamma * np.max(future_quality, axis=1)
        return targets
    
    def _improve_strategy_standard(self, gamma, b_state, b_action, b_reward, b_next_state, b_done):
        batch_size = b_state.shape[0]
        # standard strategy:   dE(w)/dw = (Qw(s,a) - (r + y * max_A Qw(s',A)) dQw(s,a)/dw
        #                      w <- w - alpha * dE(w)/dw
        future_quality = self.q_function(b_next_state)
        # set required reward and consider next state if not terminal
        targets = self.q_function(b_state)
        targets[np.arange(batch_size), b_action] = \
            b_reward + np.logical_not(b_done) * gamma * np.max(future_quality, axis=1)
        return targets
    
    def _improve_target_network(self, tau=0.005):
        model_weights = self.q_function.get_weights()
        target_weights = self.q_function_target.get_weights()
        for i in range(len(model_weights)):
            target_weights[i] = tau * model_weights[i] + (1 - tau) * target_weights[i]
        self.q_function_target.set_weights(target_weights)
    
    def train(self, verbose: int = None, render=False):
        """
        Performs Deep Q-Learning algorithm for the given environment

        Args:
            verbose: level of the printed debug info
            render: whether to render the environment at each step

        Returns:
            reward_list: 1-d dimensional array of the overall reward obtained in each episode
        """
        verbose = 0 if not verbose else int(verbose)
        
        # whether EPSILON_DECAY is a factor or a desired maximum number of episodes after which
        # epsilon becomes EPSILON_FINAL
        epsilon_decay = EPSILON_DECAY if EPSILON_DECAY < 1 \
            else (EPSILON_FINAL / EPSILON_INIT) ** (1 / EPSILON_DECAY)
        epsilon = EPSILON_INIT
        reward_list = []
        
        time_start = perf_counter()
        
        for episode in range(1, TOT_EPISODES + 1):
            
            # reset environment
            state = self.state_normalizer(self.env.reset())
            episode_reward = 0
            episode_steps = 0
            done = False
            
            #
            # SIMULATE AND IMPROVE
            #
            while not done:
                # CHOOSE STATE and STEP
                # epsilon greedy implementation (improves exploration)
                pred_action, pred_q_value = self._epsilon_greedy_best_action(state, epsilon)
                # go to next state with selected action
                next_state, reward, done, _ = self.env.step(pred_action)
                next_state = self.state_normalizer(next_state)
                episode_reward += reward
                episode_steps += 1
                
                if render:
                    self.env.render()
                
                # IMPROVE NETWORK
                # update experience buffer with what I've learned and improve the Q function
                self.exp_buffer.append([state, pred_action, reward, next_state, done])
                if len(self.exp_buffer) >= MIN_OBSERVATION:
                    self._improve_network(MINIBATCH_SIZE, GAMMA)
                    self._improve_target_network(TAU)
                
                # move on to the next state
                state = next_state
            
            # episode ended
            if render:
                self.env.close()
            epsilon = max(epsilon * epsilon_decay, EPSILON_FINAL)
            reward_list.append(episode_reward)
            if verbose > 0:
                # noinspection PyTypeChecker
                self.save_rewards(reward_list)
            
            #
            # EPISODE RESULTS
            #
            
            # checkpoint
            if verbose > 0:
                print(f"Episode {episode:7d}   -->   "
                      f"reward: {episode_reward:8.2f}  (in {episode_steps:4d} steps)   "
                      f"epsilon: {epsilon:4.2f}   "
                      f"time: {get_time(time_start)}   "
                      f"recent_rewards_mean: {np.mean(reward_list[-100:]):8.2f}")
            if episode % 100 == 0:
                self.save_checkpoint(verbose=verbose)
        
        if verbose > 0:
            print("Training concluded")
            print(f" - total elapsed time: {get_time(time_start)}")
        
        self.save_checkpoint(name="FINAL", verbose=verbose)
        # noinspection PyTypeChecker
        self.save_rewards(reward_list)
        
        return reward_list
    
    def predict(self, state):
        state = self.state_normalizer(state)
        actions_quality = self.q_function(state)
        pred_action = actions_quality.argmax()
        return pred_action
    
    def create_model(self):
        """
        Create a DNN model for the discrete Aqua environment

        Returns:
            model: the corresponding neural network
        """
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        hidden_0 = tf.keras.layers.Dense(units=64, activation='relu')(inputs)
        hidden_1 = tf.keras.layers.Dense(units=64, activation='relu')(hidden_0)
        outputs = tf.keras.layers.Dense(self.action_space, activation='linear')(hidden_1)
        model = tf.keras.Model(inputs, outputs)
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model
    
    def save_network(self, name=None, verbose=None):
        idx = name if name else f"{self.checkpoint_counter:05d}"
        filepath = os.path.join(self.checkpoint_directory, f"{WEIGHTS_PREFIX}-{idx}.h5")
        self.q_function.save_weights(filepath)
        if verbose > 0:
            print(f"Saved network to disk - {self.checkpoint_counter}")
    
    def load_network(self, checkpoint_num):
        idx = checkpoint_num if isinstance(checkpoint_num, str) else f"{checkpoint_num:05d}"
        filepath = os.path.join(self.checkpoint_directory, f"{WEIGHTS_PREFIX}-{idx}.h5")
        self.q_function.load_weights(filepath)
        print(f"Loaded network from disk - {checkpoint_num}")
    
    # TODO TypeError: Object of type ndarray is not JSON serializable
    def save_buffer(self, name=None, verbose=None):
        idx = name if name else f"{self.checkpoint_counter:5d}"
        filepath = os.path.join(self.checkpoint_directory, f"{EXP_BUFFER_PREFIX}-{idx}.json")
        with open(filepath, 'w') as exp_buff_file:
            dump(list(self.exp_buffer), exp_buff_file, indent='\t')
        if verbose > 0:
            print(f"Saved experiences to disk - {self.checkpoint_counter}")
    
    def load_buffer(self, checkpoint_num):
        idx = checkpoint_num if isinstance(checkpoint_num, str) else f"{checkpoint_num:05d}"
        filepath = f"{EXP_BUFFER_PREFIX}-{idx}.json"
        with open(os.path.join(self.checkpoint_directory, filepath), 'r') as exp_buff_file:
            self.exp_buffer = deque(load(exp_buff_file), maxlen=BUFFER_SIZE)
        print(f"Loaded experiences from disk - {checkpoint_num}")
    
    def save_checkpoint(self, name=None, verbose=None):
        self.save_network(name=name, verbose=verbose)
        # self.save_buffer(name=name, verbose=verbose) TODO restore this, see above
        self.checkpoint_counter += 1
    
    def load_checkpoint(self, num: Union[int, str] = None):
        prefix = f"{WEIGHTS_PREFIX}-"
        ids = [f.replace(prefix, "").replace(".h5", "")
               for f in os.listdir(self.checkpoint_directory)
               if f.startswith(prefix) and f.endswith(".h5")]
        num_ids = [num_id for num_id in ids if num_id.isnumeric()]
        load_checkpoint = max(num_ids)
        self.checkpoint_counter = load_checkpoint
        if num is not None:
            load_checkpoint = num
            if isinstance(num, int):
                self.checkpoint_counter = num
        
        self.load_network(load_checkpoint)
        # self.load_buffer(load_checkpoint) TODO restore this, see above
    
    def save_rewards(self, reward_list):
        np.savetxt(os.path.join(self.working_directory, "rewards.txt"), reward_list)


class DQN_alt:
    def __init__(self, env, verbose, normalizer=None):
        self.env = env
        self.verbose = verbose
        self.state_normalizer = (lambda x: x) if not normalizer else normalizer
        
        self.input_shape = self.env.observation_space.shape
        self.action_space = env.action_space.n
        self.actor = self.get_actor_model()
        self.actor_target = self.get_actor_model()
        self.actor_target.set_weights(self.actor.get_weights())
        
        self.optimizer = tf.keras.optimizers.Adam()
        self.gamma = 0.98
        self.memory_size = 50000
        self.batch_size = 64
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995
        self.tau = 0.005
        
        self.run_id = np.random.randint(0, 1000)
        self.render = False
    
    def loop(self, num_episodes=1000):
        reward_list = []
        ep_reward_mean = deque(maxlen=100)
        replay_buffer = deque(maxlen=self.memory_size)
        
        for episode in range(num_episodes):
            state = self.env.reset()
            state = self.state_normalizer(state)
            ep_reward = 0
            
            while True:
                if self.render:
                    self.env.render()
                action = self.get_action(state)
                new_state, reward, done, _ = self.env.step(action)
                new_state = self.state_normalizer(new_state)
                ep_reward += reward
                
                replay_buffer.append([state, action, reward, new_state, done])
                if done:
                    break
                state = new_state
                
                self.update_networks(replay_buffer)
                self._update_target(tau=self.tau)
            
            self.exploration_rate = max(self.exploration_rate * self.exploration_decay, 0.05)
            ep_reward_mean.append(ep_reward)
            reward_list.append(ep_reward)
            if self.verbose > 0:
                print(f"Episode: {episode:7.0f}, "
                      f"reward: {ep_reward:8.2f}, "
                      f"mean_last_100: {np.mean(ep_reward_mean):8.2f}, "
                      f"exploration: {self.exploration_rate:0.2f}")
            if self.verbose > 1:
                np.savetxt(f"dqn_with_obs/4/rewards.txt", reward_list)
    
    def _update_target(self, tau):
        weights, target_weights = self.actor.variables, self.actor_target.variables
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))
    
    def get_action(self, state):
        if np.random.random() < self.exploration_rate:
            return np.random.choice(self.action_space)
        return np.argmax(self.actor(state.reshape((1, -1))))
    
    def update_networks(self, replay_buffer):
        samples = np.array(random.sample(replay_buffer, min(len(replay_buffer), self.batch_size)),
                           dtype=object)
        with tf.GradientTape() as tape:
            # Compute loss with custom loss function
            objective_function = self.actor_objective_function_double(samples)
            # Compute gradients actor for network
            grads = tape.gradient(objective_function, self.actor.trainable_variables)
            # Apply gradients to update network weights
            self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
    
    def actor_objective_function_double(self, replay_buffer):
        state = np.vstack(replay_buffer[:, 0])
        action = replay_buffer[:, 1]
        reward = np.vstack(replay_buffer[:, 2])
        new_state = np.vstack(replay_buffer[:, 3])
        done = np.vstack(replay_buffer[:, 4])
        
        next_state_action = np.argmax(self.actor(new_state), axis=1)
        target_mask = self.actor_target(new_state) * tf.one_hot(next_state_action,
                                                                self.action_space)
        target_mask = tf.reduce_sum(target_mask, axis=1, keepdims=True)
        
        target_value = reward + (1 - done.astype(int)) * self.gamma * target_mask
        mask = self.actor(state) * tf.one_hot(action, self.action_space)
        prediction_value = tf.reduce_sum(mask, axis=1, keepdims=True)
        
        mse = tf.math.square(prediction_value - target_value)
        return tf.math.reduce_mean(mse)
    
    def get_actor_model(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        hidden_0 = tf.keras.layers.Dense(64, activation='relu')(inputs)
        hidden_1 = tf.keras.layers.Dense(64, activation='relu')(hidden_0)
        outputs = tf.keras.layers.Dense(self.action_space, activation='linear')(hidden_1)
        
        return tf.keras.Model(inputs, outputs)
    
    ##########################
    #### VANILLA METHODS #####
    ##########################
    
    def actor_objective_function_fixed_target(self, replay_buffer):
        state = np.vstack(replay_buffer[:, 0])
        action = replay_buffer[:, 1]
        reward = np.vstack(replay_buffer[:, 2])
        new_state = np.vstack(replay_buffer[:, 3])
        done = np.vstack(replay_buffer[:, 4])
        
        target_value = reward + (1 - done.astype(int)) * self.gamma * np.amax(
            self.actor_target(new_state), axis=1, keepdims=True)
        mask = self.actor(state) * tf.one_hot(action, self.action_space)
        prediction_value = tf.reduce_sum(mask, axis=1, keepdims=True)
        
        mse = tf.math.square(prediction_value - target_value)
        return tf.math.reduce_mean(mse)
    
    def actor_objective_function_std(self, replay_buffer):
        state = np.vstack(replay_buffer[:, 0])
        action = replay_buffer[:, 1]
        reward = np.vstack(replay_buffer[:, 2])
        new_state = np.vstack(replay_buffer[:, 3])
        done = np.vstack(replay_buffer[:, 4])
        
        target_value = reward + (1 - done.astype(int)) * self.gamma * np.amax(self.actor(new_state),
                                                                              axis=1, keepdims=True)
        mask = self.actor(state) * tf.one_hot(action, self.action_space)
        prediction_value = tf.reduce_sum(mask, axis=1, keepdims=True)
        
        mse = tf.math.square(prediction_value - target_value)
        return tf.math.reduce_mean(mse)


class DQNAquaPolicy(AquaPolicy):
    class AquaStateNormalizer(object):
        
        def __init__(self, env):
            self.env = env
            high_obs = self.env.observation_space.high
            low_obs = self.env.observation_space.low
            self.observation_range = high_obs - low_obs
        
        def __call__(self, state):
            state /= self.observation_range  # set coordinates in [0,1] and angle in [-0.5,0.5]
            state[2] += 0.5  # set angle in [0,1]
            return state[np.newaxis, :]
        
        def inverse(self, state):
            state[0, 2] -= 0.5
            state *= self.observation_range
            return state[0, :]
    
    def __init__(self, folder, load_network, params):
        super().__init__(is_discrete=True, params=params)
        
        # get the DQN implementation
        self.dqn = DQN(self.env, folder, normalizer=DQNAquaPolicy.AquaStateNormalizer(self.env))
        if load_network is not None:
            if isinstance(load_network, bool):
                if load_network:
                    self.dqn.load_checkpoint()
            else:
                self.dqn.load_checkpoint(load_network)
    
    def train(self, debug, render):
        return self.dqn.train(verbose=debug, render=render)
    
    def get_action(self, state):
        return self.dqn.predict(state)


class DQNAquaPolicy_alt(AquaPolicy):
    class AquaStateNormalizer(object):
        
        def __init__(self, env):
            self.env = env
            high_obs = self.env.observation_space.high
            low_obs = self.env.observation_space.low
            self.observation_range = high_obs - low_obs
        
        def __call__(self, state):
            state /= self.observation_range  # set coordinates in [0,1] and angle in [-0.5,0.5]
            state[2] += 0.5  # set angle in [0,1]
            return state[np.newaxis, :]
        
        def inverse(self, state):
            state[0, 2] -= 0.5
            state *= self.observation_range
            return state[0, :]
    
    def __init__(self, folder, load_network, params):
        super().__init__(is_discrete=True, params=params)
        
        # get the DQN implementation
        self.dqn = DQN_alt(self.env, verbose=2, normalizer=DQNAquaPolicy.AquaStateNormalizer(self.env))
    
    def train(self, debug, render):
        return self.dqn.loop(num_episodes=TOT_EPISODES)
    
    def get_action(self, state):
        return self.dqn.get_action(state)
