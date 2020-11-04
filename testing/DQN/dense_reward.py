import gym
from keras.engine.saving import model_from_json

import gym_aqua
import numpy as np
import os

from testing.DQN.network import DQN, create_model


def to_infinity():
    time = 0
    while True:
        time += 1
        yield time


if __name__ == '__main__':
    
    # we register the environments in gym_aqua/__init__.py
    # so now we're ready
    
    env = gym.make('AquaSmall-v1')
    env.reset()
    
    # action space is continuous, but we can't handle it -> use discrete tuple
    action_space_resolution = 5
    action_space_discrete = np.linspace(env.action_space.low, env.action_space.high,
                                        num=action_space_resolution, endpoint=True)
    actions = [(aL, aR) for aL in action_space_discrete[:, 0] for aR in action_space_discrete[:, 1]]
    
    #TODO
    # rete neurale
    # report (4 pagine)

    if not os.path.exists("model.json"):
        # 3 input per stato (x_boat,y_boat,angle_boat) !! ignoro gli ostacoli...
        # 25 output (una per azione)
        neural_network = create_model(3, len(actions), 32, 5)
        neural_network, score = DQN(env, actions, neural_network, trials=100, max_iter=4000)
    
        # save results
        with open("model.json", "w") as json_file:
            json_file.write(neural_network.to_json())
        neural_network.save_weights("model.h5")
        print("Saved model to disk")

    # load json and create model
    with open('model.json', 'r') as json_file:
        loaded_model = model_from_json(json_file.read())
    loaded_model.load_weights("model.h5")
    loaded_model.compile(loss="mean_squared_error", optimizer='adam')
    print("Loaded model from disk")
    
    # testing
    state = env.reset()
    done = False
    reward = 0
    for i in to_infinity():
        env.render()

        if done:
            if reward < 0:
                print("schiantati...")
            else:
                print("goal!")
            break

        state = state[np.newaxis, :]
        action = loaded_model.predict(state)
        state, reward, done, _ = env.step(actions[action.argmax()])

    env.close()
