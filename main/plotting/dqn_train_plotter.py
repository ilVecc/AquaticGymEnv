from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


class ProgressPlotter(object):
    
    def __init__(self, fig_size=(12, 12)):
        self._store = {}
        self._figure_size = fig_size
    
    def plot_training_success(self, save=False, window=200):
        
        fig, ax = plt.subplots(1, figsize=self._figure_size)
        # seaborn "darkgrid" style
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_facecolor('#eaeaf2')
        plt.grid(color='#ffffff', linestyle='-', linewidth=1)
        plt.xticks(fontsize=16)
        plt.yticks(range(0, 101, 10), fontsize=16)
        ax.set_yticklabels([f"{i}%" for i in range(0, 101, 10)])
        ax.set_ylim(-5, 105)
        plt.xlabel("Episodes", fontsize=16)
        plt.ylabel("Success ratio", fontsize=16)
        plt.title(f"Aqua success (window of {window} episodes)", fontsize=24)
        
        for label, policies_info in self._store.items():
            # fetch data
            n_episodes = policies_info["n_episodes"]
            
            # prepare raw data
            success = np.vstack([policy_rewards[:n_episodes] for policy_rewards in policies_info["success"]])  # runs X episodes
            success = np.hstack([np.zeros(shape=(policies_info["n_policies"], window - 1)), success])  # add padding
            
            success = np.hstack([(success[:, i:i + window] > 0).mean(axis=1, keepdims=True)
                                 for i in range(n_episodes)]) * 100
            
            # prepare plot data
            episodes = np.arange(n_episodes) + 1
            success_avg = success.mean(axis=0)
            success_std = success.std(axis=0)
            
            # plot data (colors are automatically paired thanks to plt under the hood magic)
            plt.plot(episodes, success_avg, label=label)
            plt.fill_between(episodes, success_avg + success_std, success_avg - success_std, alpha=0.3)
        
        plt.legend(title="Env Type", loc="upper left")
        
        if save:
            plots_path = Path('plots')
            plots_path.mkdir(parents=True, exist_ok=True)
            if len(self._store) == 1:
                # just one env, reuse already loaded data
                if policies_info["n_policies"] == 1:
                    # just one policy
                    plot_policy_path = plots_path / Path(policies_info["policies"][0])
                    plot_policy_path.mkdir(parents=True, exist_ok=True)
                    plt.savefig(str(plot_policy_path / "training_policy_success.png"))
                else:
                    # more than one policy
                    plot_policy_path = plots_path / Path(policies_info["policies"][0]).parent
                    plot_policy_path.mkdir(parents=True, exist_ok=True)
                    plt.savefig(str(plot_policy_path / "training_env_success.png"))
            else:
                plt.savefig(str(plots_path / "training_overall_success.png"))
        plt.show()
    
    def plot_training_rewards(self, save=False, window=200):
        fig, ax = plt.subplots(1, figsize=self._figure_size)
        # seaborn "darkgrid" style
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_facecolor('#eaeaf2')
        plt.grid(color='#ffffff', linestyle='-', linewidth=1)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel("Episodes", fontsize=16)
        plt.ylabel("Rewards", fontsize=16)
        plt.title(f"Aqua rewards (window of {window} episodes)", fontsize=24)
        
        for label, policies_info in self._store.items():
            # fetch data
            n_episodes = policies_info["n_episodes"]
            
            # prepare raw data
            rewards = np.vstack([policy_rewards[:n_episodes] for policy_rewards in policies_info["rewards"]])  # runs X episodes
            rewards = np.hstack([np.zeros(shape=(policies_info["n_policies"], window - 1)), rewards])  # add padding
            
            rewards = np.hstack([rewards[:, i:i + window].sum(axis=1, keepdims=True) / min(i + 1, window)
                                 for i in range(n_episodes)])
            
            # prepare plot data
            episodes = np.arange(n_episodes) + 1
            rewards_avg = rewards.mean(axis=0)
            rewards_std = rewards.std(axis=0)
            
            # plot data (colors are automatically paired thanks to plt under the hood magic)
            plt.plot(episodes, rewards_avg, label=label)
            plt.fill_between(episodes, rewards_avg + rewards_std, rewards_avg - rewards_std, alpha=0.3)
        
        plt.legend(title="Env Type", loc="upper left")
        
        if save:
            plots_path = Path('plots')
            plots_path.mkdir(parents=True, exist_ok=True)
            if len(self._store) == 1:
                # just one env, reuse already loaded data
                if policies_info["n_policies"] == 1:
                    # just one policy
                    plot_policy_path = plots_path / Path(policies_info["policies"][0])
                    plot_policy_path.mkdir(parents=True, exist_ok=True)
                    plt.savefig(str(plot_policy_path / "training_policy_rewards.png"))
                else:
                    # more than one policy
                    plot_policy_path = plots_path / Path(policies_info["policies"][0]).parent
                    plot_policy_path.mkdir(parents=True, exist_ok=True)
                    plt.savefig(str(plot_policy_path / "training_env_rewards.png"))
            else:
                plt.savefig(str(plots_path / "training_overall_rewards.png"))
        plt.show()
    
    def load(self, label, policy_paths: List[str], fetch=True):
        
        # collect data
        rewards_path = [str(Path(policy_path) / "rewards.txt") for policy_path in policy_paths]
        success_path = [str(Path(policy_path) / "success.txt") for policy_path in policy_paths]
        
        rewards = [lambda path=path: np.loadtxt(path) for path in rewards_path]  # "currying", cool hack: lambdas are evaluated dynamically, so just last value
        success = [lambda path=path: np.loadtxt(path) for path in success_path]  # of "path" is used... but default values are evaluated immediately!
        min_size = None
        
        if fetch:
            rewards = [loader() for loader in rewards]
            success = [loader() for loader in success]
            min_size = min([len(policy_rewards) for policy_rewards in rewards])
        
        self._store[label] = {
            "policies": policy_paths,
            "n_policies": len(policy_paths),
            "n_episodes": min_size,
            "rewards": rewards,
            "success": success
        }


if __name__ == "__main__":
    plotter = ProgressPlotter(fig_size=(12, 8))
    plotter.load("No obstacles", ["example_policies/example_no_obs"], fetch=True)
    plotter.load("With obstacles", ["example_policies/example_with_obs"], fetch=True)
    plotter.plot_training_rewards(save=True, window=500)
    plotter.plot_training_success(save=True, window=500)
