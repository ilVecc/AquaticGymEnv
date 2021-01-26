import os
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
from collections import deque

import numpy as np
import tensorflow as tf

# allow multiple trainings in parallel
for gpu_instance in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu_instance, True)
import datetime as dt
from time import perf_counter

default_hyperparam = {
    "BUFFER_SIZE": 50000,
    
    "EPSILON_FINAL": 0.05,
    "EPSILON_INIT": 1.0,
    "EPSILON_DECAY": 10000,  # either a factor or a number of episodes
    
    "MINIBATCH_SIZE": 64,
    "GAMMA": 0.98,
    "TAU": 0.005,
    
    "LOSS_STRATEGY": "double",  # / fixed / standard
    "IMPROVE_STRATEGY": "custom_grad"  # / standard
}


#
# IMPLEMENTATION DETAILS
#
# Tensorflow 2.x better handles prediction on small batches using __call__(); use predict() only
# when using big batches. Since here we always ask for either 1 or MINIBATCH_SIZE predictions,
# we will be using __call__()
#
# You can check by yourself that doing so saves a ton of time
#

def get_time(start):
    return str(dt.timedelta(seconds=int(perf_counter() - start)))


class Normalizer(object):
    def __call__(self, *args, **kwargs):
        pass
    
    def inverse(self, norm_state):
        pass


class DQN:
    
    def __init__(self, environment, save_directory, save_freq=100, normalizer: Normalizer = None, hyperparams=None):
        """
        Initialize a DQN instance.

        Args:
            environment: the OpenAI Gym environment
            save_directory: where to save the results and the checkpoint models
            save_freq: number of episodes between each checkpoint
            normalizer: a Normalizer object to be applied to each state before passing it to the network
            hyperparams: custom dictionary of hyperparameters for each step of the learning, the default can be found at the beginning of the file
        """
        # hyper-parameters and constants
        if hyperparams is None:
            hyperparams = {}
        self.BUFFER_SIZE = hyperparams["BUFFER_SIZE"] if "BUFFER_SIZE" in hyperparams else default_hyperparam["BUFFER_SIZE"]
        
        self.EPSILON_FINAL = hyperparams["EPSILON_FINAL"] if "EPSILON_FINAL" in hyperparams else default_hyperparam["EPSILON_FINAL"]
        self.EPSILON_INIT = hyperparams["EPSILON_INIT"] if "EPSILON_INIT" in hyperparams else default_hyperparam["EPSILON_INIT"]
        self.EPSILON_DECAY = hyperparams["EPSILON_DECAY"] if "EPSILON_DECAY" in hyperparams else default_hyperparam["EPSILON_DECAY"]
        
        self.MINIBATCH_SIZE = hyperparams["MINIBATCH_SIZE"] if "MINIBATCH_SIZE" in hyperparams else default_hyperparam["MINIBATCH_SIZE"]
        self.GAMMA = hyperparams["GAMMA"] if "GAMMA" in hyperparams else default_hyperparam["GAMMA"]
        self.TAU = hyperparams["TAU"] if "TAU" in hyperparams else default_hyperparam["TAU"]
        self.LOSS_STRATEGY = hyperparams["LOSS_STRATEGY"] if "LOSS_STRATEGY" in hyperparams else default_hyperparam["LOSS_STRATEGY"]
        if self.LOSS_STRATEGY == "double":
            self.LOSS_STRATEGY = self._improve_strategy_double
        elif self.LOSS_STRATEGY == "fixed":
            self.LOSS_STRATEGY = self._improve_strategy_fixed
        elif self.LOSS_STRATEGY == "standard":
            self.LOSS_STRATEGY = self._improve_strategy_standard
        else:
            raise NotImplementedError(f"no strategy \"{self.LOSS_STRATEGY}\" implemented")
        self.IMPROVE_STRATEGY = hyperparams["IMPROVE_STRATEGY"] if "IMPROVE_STRATEGY" in hyperparams else default_hyperparam["IMPROVE_STRATEGY"]
        if self.IMPROVE_STRATEGY == "standard":
            self.IMPROVE_STRATEGY = self._improve_network
        elif self.IMPROVE_STRATEGY == "custom_grad":
            self.IMPROVE_STRATEGY = self._improve_network_gradient
        else:
            raise NotImplementedError(f"no improvement \"{self.IMPROVE_STRATEGY}\" implemented")
        
        # environment related
        self.env = environment
        self.input_shape = self.env.observation_space.shape
        self.action_space = self.env.action_space.n
        # learning related
        self.q_function = self.create_model()
        self.q_function_target = self.create_model()
        self.q_function_target.set_weights(self.q_function.get_weights())
        self.exp_buffer = deque(maxlen=self.BUFFER_SIZE)
        self.state_normalizer = (lambda x: x[np.newaxis, ...]) if not normalizer else normalizer
        
        # check folder
        self.savepath = Path(save_directory)
        self.savepath.mkdir(parents=True, exist_ok=True)
        self.models_path = self.savepath / "models"
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.save_freq = save_freq
        
        self.episode = 0
        self.reward_list = []
        self.success_list = []
    
    def train(self, episodes, verbose: int = None, render=False):
        """
        Performs Deep Q-Learning algorithm for the given environment

        Args:
            episodes: number of episodes for the training
            verbose: level of the printed debug info
            render: whether to render the environment at each step

        Returns:
            reward_list: 1-d dimensional array of the overall reward obtained in each episode
        """
        
        if verbose > 0:
            print("-" * 154)
            print(f"{f'Training {self.savepath}':^154}")
            print("-" * 154)
        
        # whether EPSILON_DECAY is a factor or a desired maximum number of episodes after which
        # epsilon becomes EPSILON_FINAL
        epsilon_decay = self.EPSILON_DECAY if self.EPSILON_DECAY < 1 \
            else (self.EPSILON_FINAL / self.EPSILON_INIT) ** (1 / self.EPSILON_DECAY)
        epsilon = self.EPSILON_INIT
        self.reward_list = []
        self.success_list = []
        
        time_start = perf_counter()
        
        for self.episode in range(1, episodes + 1):
            
            # reset environment
            state = self.state_normalizer(self.env.reset())
            episode_reward = 0
            episode_steps = 0
            info = {}
            done = False
            
            #
            # SIMULATE AND IMPROVE
            #
            while not done:
                # CHOOSE STATE and STEP
                # epsilon greedy implementation (improves exploration)
                pred_action, pred_q_value = self._epsilon_greedy_best_action(state, epsilon)
                # go to next state with selected action
                next_state, reward, done, info = self.env.step(pred_action)
                next_state = self.state_normalizer(next_state)
                episode_reward += reward
                episode_steps += 1
                
                if render:
                    self.env.render()
                
                # IMPROVE NETWORK
                # update experience buffer with what I've learned and improve the Q function
                self.exp_buffer.append([state, pred_action, reward, next_state, done])
                self.IMPROVE_STRATEGY(self.MINIBATCH_SIZE, self.GAMMA)
                self._improve_target_network(self.TAU)
                
                # move on to the next state
                state = next_state
            
            # episode ended
            if render:
                self.env.close()
            epsilon = max(epsilon * epsilon_decay, self.EPSILON_FINAL)
            self.reward_list.append(episode_reward)
            self.success_list.append(info['Termination.success'])
            
            #
            # EPISODE RESULTS
            #
            
            # checkpoint
            if verbose:
                print(f"Episode {self.episode:7d}   -->   "
                      f"reward: {episode_reward:8.2f}  (in {episode_steps:4d} steps)   "
                      f"mean_last_100: {np.mean(self.reward_list[-100:]):8.2f}   "
                      f"success: {str(self.success_list[-1]):>5}   "
                      f"mean_last_100: {np.mean(self.success_list[-100:]) * 100:3.0f}%   "
                      f"epsilon: {epsilon:4.2f}   "
                      f"time: {get_time(time_start)}")
            if self.episode % self.save_freq == 0:
                self.save_progress()
        
        if verbose:
            print("Training concluded")
            print(f" - total elapsed time: {get_time(time_start)}")
        
        self.save_progress()
        
        return self.reward_list, self.success_list
    
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
    
    def _improve_network(self, minibatch_size, gamma):
        """Update targets using batch prediction/training for each minibatch, so each subsequent
        minibatches predictions are made upon the updates of the previous minibatches"""
        b_state, b_action, targets = self.LOSS_STRATEGY(minibatch_size, gamma)
        
        # back-propagation
        return self.q_function.train_on_batch(b_state, targets)
    
    def _improve_network_gradient(self, minibatch_size, gamma):
        with tf.GradientTape() as tape:
            b_state, b_action, target_values = self.LOSS_STRATEGY(minibatch_size, gamma)
            
            # get current prediction for each state/action pair in the batch
            mask = self.q_function(b_state) * tf.one_hot(b_action, self.action_space)
            prediction_value = tf.reduce_sum(mask, axis=1, keepdims=True)
            
            mse = tf.math.square(prediction_value - target_values)
            objective_function = tf.math.reduce_mean(mse)  # Compute loss with custom loss function
            grads = tape.gradient(objective_function, self.q_function.trainable_variables)  # Compute gradients actor for network
            self.q_function.optimizer.apply_gradients(zip(grads, self.q_function.trainable_variables))  # Apply gradients to update network weights
    
    def _sample_experience(self, minibatch_size):
        batch_size = min(len(self.exp_buffer), minibatch_size)
        batch = list(zip(*random.sample(self.exp_buffer, batch_size)))
        # maps the tuples (s, a, r, sn, d) in the buffer to a tuple of np.ndarrays
        b_state = np.vstack(batch[0])
        b_action = np.array(batch[1])
        b_reward = np.array(batch[2])
        b_next_state = np.vstack(batch[3])
        b_done = np.array(batch[4])
        return b_state, b_action, b_reward, b_next_state, b_done, batch_size
    
    def _improve_strategy_double(self, minibatch_size, gamma):
        b_state, b_action, b_reward, b_next_state, b_done, b_size = self._sample_experience(minibatch_size)
        # double strategy:     dE(w)/dw = (Qw(s,a) - (r + y * Qwt(s',a')) dQw(s,a)/dw
        #                                                      with  a' = argmax_A Qw(s',A)
        #                      w <- w - alpha * dE(w)/dw
        best_next_actions = np.argmax(self.q_function(b_state).numpy(), axis=1)
        future_quality = self.q_function_target(b_next_state).numpy()[np.arange(b_size), best_next_actions]
        # set required reward and consider next state if not terminal
        targets = self.q_function(b_state).numpy()
        targets[np.arange(b_size), b_action] = b_reward + np.logical_not(b_done) * gamma * future_quality
        return b_state, b_action, targets
    
    def _improve_strategy_fixed(self, minibatch_size, gamma):
        b_state, b_action, b_reward, b_next_state, b_done, b_size = self._sample_experience(minibatch_size)
        # fixed strategy:      dE(w)/dw = (Qw(s,a) - (r + y * max_A Qwt(s',A)) dQw(s,a)/dw
        #                      w <- w - alpha * dE(w)/dw
        future_quality = self.q_function_target(b_next_state)
        # set required reward and consider next state if not terminal
        targets = self.q_function(b_state)
        targets[np.arange(b_size), b_action] = b_reward + np.logical_not(b_done) * gamma * np.max(future_quality, axis=1)
        return b_state, b_action, targets
    
    def _improve_strategy_standard(self, minibatch_size, gamma):
        b_state, b_action, b_reward, b_next_state, b_done, b_size = self._sample_experience(minibatch_size)
        # standard strategy:   dE(w)/dw = (Qw(s,a) - (r + y * max_A Qw(s',A)) dQw(s,a)/dw
        #                      w <- w - alpha * dE(w)/dw
        future_quality = self.q_function(b_next_state)
        # set required reward and consider next state if not terminal
        targets = self.q_function(b_state)
        targets[np.arange(b_size), b_action] = b_reward + np.logical_not(b_done) * gamma * np.max(future_quality, axis=1)
        return b_state, b_action, targets
    
    def _improve_target_network(self, tau):
        model_weights = self.q_function.get_weights()
        target_weights = self.q_function_target.get_weights()
        for i in range(len(model_weights)):
            target_weights[i] = tau * model_weights[i] + (1 - tau) * target_weights[i]
        self.q_function_target.set_weights(target_weights)
    
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
    
    def save_progress(self):
        # noinspection PyTypeChecker
        np.savetxt(str(self.savepath / "rewards.txt"), self.reward_list)
        # noinspection PyTypeChecker
        np.savetxt(str(self.savepath / "success.txt"), self.success_list)
        self.q_function.save(str(self.models_path / f"model-{self.episode // self.save_freq:05d}"))
