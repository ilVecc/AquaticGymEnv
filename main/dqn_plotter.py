import os
from typing import Union, List

import matplotlib.pyplot as plt
import numpy as np


def plot_results(policy_ids: Union[str, List[str]], labels=None, colors=None,
                 run_selector=None, save=False, window=200):
        
    fig, ax = plt.subplots(1, figsize=(14, 8))
    # seaborn "darkgrid" style
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_facecolor('#eaeaf2')
    plt.grid(color='#ffffff', linestyle='-', linewidth=1)
    plt.xticks(fontsize=12)
    plt.yticks(range(0, 101, 10), fontsize=12)
    ax.set_yticklabels([f"{i}%" for i in range(0, 101, 10)])
    ax.set_ylim(-5, 105)

    if isinstance(policy_ids, str):
        policy_ids = [policy_ids]
        
    for i in range(len(policy_ids)):
        policy_id = policy_ids[i]
        rewards = [np.loadtxt(os.path.join(policy_id, run_id, "rewards.txt"))
                   for run_id in os.listdir(policy_id)
                   if os.path.splitext(run_id)[1] == ''
                   and (run_selector is None or run_id == str(run_selector))]
        min_size = min([len(run_scores) for run_scores in rewards])
        n_runs = len(rewards)
        n_episodes = min_size

        episodes = np.arange(n_episodes) + 1
        rewards = np.vstack([run_scores[:min_size] for run_scores in rewards])
        rewards = np.hstack([np.zeros(shape=(n_runs, window - 1)), rewards])  # add padding
        
        success = np.array([(rewards[:, i:i + window] > 0).sum(axis=1) / window
                            for i in range(n_episodes)]) * 100
        success_mean = success.mean(axis=1)
        success_std = success.std(axis=1)

        plt.plot(episodes, success_mean, color=colors[i], label=labels[i])
        plt.fill_between(episodes, success_mean + success_std, success_mean - success_std,
                         facecolor=colors[i], alpha=0.3)

    plt.xlabel("Episodes", fontsize=12)
    plt.ylabel("Success ratio", fontsize=12)
    plt.title(f"Aqua success (window of {window} episodes)", fontsize=18)
    plt.legend(title="Env Type", loc="lower right")

    if save:
        if len(policy_ids) == 1:
            policy_id = policy_ids[0]
            if run_selector is not None:
                plt.savefig(os.path.join(policy_id, str(run_selector), "success.png"))
            else:
                plt.savefig(os.path.join(policy_id, "overall_success.png"))
        else:
            plt.savefig(f"comparison_success.png")
    plt.show()


def plot_qvalue(size=100, sectors=8):
    x = np.tile((np.arange(1, size + 1) / size).repeat(sectors), (size, 1))
    a = np.tile(np.linspace([0], [1], sectors, endpoint=False), (size ** 2, 1))
    g = np.tile(np.array([[15, 65]]) / size, (sectors * size ** 2, 1))
    zz = np.concatenate((x.reshape([-1, 1]), x.transpose().reshape([-1, 1]), a, g), axis=1)
    
    import tensorflow.keras as keras
    inputs = keras.layers.Input(shape=5)
    hidden_0 = keras.layers.Dense(units=64, activation='relu')(inputs)
    hidden_1 = keras.layers.Dense(units=64, activation='relu')(hidden_0)
    outputs = keras.layers.Dense(4, activation='linear')(hidden_1)
    model = keras.Model(inputs, outputs)
    model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=0.001))
    model.load_weights("./dqn_fixgoal_obs/checkpoint/checkpoint-00134.h5")
    
    q_val = model.predict(zz)
    
    best_action = np.flipud(q_val.argmax(axis=1).reshape([size, size, sectors]))
    best_q_val = np.flipud(q_val.max(axis=1).reshape([size, size, sectors]))
    
    for i in range(sectors):
        # TODO improve this
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex='all', sharey='all')
        ax1.imshow(best_q_val[:, :, i])
        ax1.set_title("Best q-value")
        ax2.imshow(best_action[:, :, i])
        ax2.set_title("Best action")
        fig.suptitle("Angle {:.2f}°".format(360 * i / sectors - 180))
        fig.show()


if __name__ == "__main__":
    # plot_results(["dqn_no_obs", "dqn_with_obs"],
    #              labels=["No obstacles", "With obstacles"],
    #              colors=["red", "green"],
    #              run_selector=None, save=True)
    plot_results(["dqn_with_obs"],
                 labels=["No obstacles"],
                 colors=["red"],
                 run_selector=4, save=True)
