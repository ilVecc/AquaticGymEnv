import random
from collections import deque

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense


def create_model(input_size, output_size, hidden_layer_size, hidden_layer_number):
    """
    Create the neural netowrk model with the given parameters

    Args:
        input_size: the number of nodes for the input layer
        output_size: the number of nodes for the output layer
        hidden_layer_size: the number of nodes for each hidden layer
        hidden_layer_number: the number of hidden layers

    Returns:
        model: the corresponding neural network
    """
    model = Sequential()
    # input layer + hidden layer #1
    model.add(Dense(hidden_layer_number, input_dim=input_size, activation="relu"))
    # add all the other hidden layers (one it's been already created)
    for _ in range(hidden_layer_number - 1):
        model.add(Dense(hidden_layer_size, activation="relu"))  # hidden layer #i
    # output layer
    model.add(Dense(output_size, activation="linear"))
    
    model.compile(loss="mean_squared_error",
                  optimizer='adam')  # loss function and optimizer definition
    return model


def train_model(neural_network, memory, gamma=0.95):
    """
    Performs the value iteration algorithm for a specific environment

    Args:
        neural_network: the neural network model to train
        memory: the memory array on which perform the training
        gamma: gamma value, the discount factor for the Bellman equation
    """
    batch_size = 32
    if len(memory) < 32:
        return neural_network
    
    batch = random.sample(memory, batch_size)
    for state, action, next_state, reward, done in batch:
        # output is [prob_left, prob_right], max of these is the best action now
        target = neural_network.predict(state)[0]
        if done:
            # there's no next state
            target[action] = reward
        else:
            # must consider next state
            max_q = max(neural_network.predict(next_state)[0])
            target[action] = reward + (gamma * max_q)
        # back-propagation
        neural_network.fit(state, target[np.newaxis, :], verbose=0)
    return neural_network


def epsilon_greedy(neural_network, state, epsilon):
    """
    Epsilon-greedy action selection function

    Args:
        neural_network: currently fitted neural network
        state: agent's current state
        epsilon: epsilon parameter

    Returns:
        action id
    """
    n_actions = neural_network.output.shape[1]
    probs = np.full(n_actions, epsilon / n_actions)
    best_action = neural_network.predict(state).argmax()
    probs[best_action] += 1 - epsilon
    return np.random.choice(n_actions, p=probs)


def DQN(environment, actions, neural_network, trials, max_iter=10000, epsilon_decay=0.995):
    """
    Performs the Q-Learning algorithm for a specific environment on a specific neural network model

    Args:
        environment: OpenAI Gym environment
        actions:
        neural_network: the neural network to train
        trials: the number of iterations for the training phase
        max_iter:
        epsilon_decay: the decay value of epsilon for the eps-greedy exploration

    Returns:
        score_queue: 1-d dimensional array of the reward obtained at each trial step
    """
    epsilon = 1.0
    epsilon_min = 0.01
    exp_buf = deque(maxlen=10000)
    score_queue = []
    
    for i in range(trials):
        # reset environment
        s = (environment.reset())[np.newaxis, :]
        episode_total_reward = 0
        iteration = 0
        while True:
            # epsilon greedy implementation
            a = epsilon_greedy(neural_network, s, epsilon)
            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            
            # go to next state with selected action
            next_state, reward, done, _ = environment.step(actions[a])
            next_state = next_state[np.newaxis, :]
            
            # update reward here
            episode_total_reward += reward
            
            # update experience buffer with what I've learned
            exp_buf.append((s, a, next_state, reward, done))
            
            # gradient descent and back-propagation
            neural_network = train_model(neural_network, exp_buf)
            
            if done:
                # reached goal state
                break
            if iteration > max_iter:
                # too many moves
                print("trial #{}: too many moves".format(i))
                break
                
            s = next_state
            iteration += 1
            
        # episode ended
        score_queue.append(episode_total_reward)
        # identify ending-state using last reward
        if reward > 0:
            # goal score reached (goal reached with less than 1000 moves)
            print("trial #{}: reached goal after {} moves".format(i, iteration))
            break
        else:
            print("trial #{}: collided after {} moves".format(i, iteration))
    
    return neural_network, score_queue
