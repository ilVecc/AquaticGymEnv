import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_network_qvalue(path, goal_state, world_size, sectors=8, save=False):
    coord_per_angle = np.tile((np.arange(1, world_size + 1) / world_size).repeat(sectors), (world_size, 1))
    angles = np.tile(np.linspace([0], [1], sectors, endpoint=False), (world_size ** 2, 1))
    goal_pos = np.tile(goal_state / world_size, (sectors * world_size ** 2, 1))
    # pos_x, pos_y, angle, goal
    states = np.concatenate((coord_per_angle.reshape([-1, 1]), coord_per_angle.transpose().reshape([-1, 1]), angles, goal_pos), axis=1)
    
    import tensorflow as tf
    model = tf.keras.models.load_model(path)
    q_val = model.predict(states)
    
    best_action = np.fliplr(np.flipud(q_val.argmax(axis=1).reshape([world_size, world_size, sectors])))
    best_q_val = np.fliplr(np.flipud(q_val.max(axis=1).reshape([world_size, world_size, sectors])))
    
    for i in range(sectors):
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex='all', sharey='all')
        fig.suptitle("Angle {:.2f}°".format(360 * i / sectors - 180))
        ax1.imshow(best_q_val[:, :, i], origin="lower")
        ax1.set_title("Best q-value")
        ax2.imshow(best_action[:, :, i], origin="lower")
        ax2.set_title("Best action")
        if save:
            savepath = Path("plots/qvalue")
            savepath.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(savepath / f"sector_{i}.png"))
        fig.show()


if __name__ == "__main__":
    
    policy_path = "example_policies/example_no_obs/models/model-00030"
    goal = np.array([[50, 50]])
    
    plot_network_qvalue(policy_path, goal_state=goal, world_size=100, save=True)
