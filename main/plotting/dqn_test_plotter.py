import sys
from pathlib import Path

import gym
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from impl.utils import AquaStateNormalizer
from testing.test_dqn import DQNAquaPolicy


class TestPlotter(object):
    
    def __init__(self, loadpath=""):
        self.results = None
        self.n_tests = 0
        if loadpath != "":
            self.results = pd.read_pickle(str(Path(loadpath) / "test_results.pickle"))
            self.n_tests = self.results.groupby("Environment").count().max().max()
    
    def run_tests(self, n_tests, savepath):
        df = pd.DataFrame(columns=["Environment", "Reward", "Success"])
        for label, tests_setup in tests_config.items():
            for policy_path in tests_setup["policies"]:
                env = gym.make("AquaEnv-v0", **tests_setup["env_params"])
                policy = DQNAquaPolicy(policy_path, normalizer=AquaStateNormalizer(env))
                for _ in tqdm(range(n_tests), desc=f"Testing: {policy_path}", file=sys.stdout):
                    tot_rew, info, tot_steps = policy.test(env, render=False)
                    df = df.append({"Environment": label, "Reward": tot_rew, "Success": int(info["Termination.success"])}, ignore_index=True)
        self.results = df
        self.n_tests = n_tests
        df["Success"] = pd.to_numeric(df["Success"])
        df.to_pickle(str(Path(savepath) / "test_results.pickle"))
    
    def plot_test_results(self, save=True):
        plt.subplots(1, figsize=(10, 10))
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel("Environment", fontsize=16)
        plt.ylabel("Reward", fontsize=16)
        plt.title(f"Aqua test rewards ({self.n_tests} runs)", fontsize=24)
        
        sns.boxplot(x="Environment", y="Reward", data=self.results)
        sns.swarmplot(x="Environment", y="Reward", data=self.results.groupby("Environment").sample(min(self.n_tests, 100)))
        if save:
            plt.savefig(str(Path("plots") / "test_overall_rewards.png"))
        plt.show()
        
        plt.subplots(1, figsize=(10, 10))
        plt.xlabel("Environment", fontsize=16)
        plt.ylabel("Success", fontsize=16)
        plt.title(f"Aqua test success ({self.n_tests} runs)", fontsize=24)
        ax = sns.countplot(data=self.results[self.results["Success"] > 0], x="Environment")
        # add percentage on top of the bars
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2., height + 3, f'{height / self.n_tests * 100:2.1f}%', ha="center")
        if save:
            plt.savefig(str(Path("plots") / "test_overall_success.png"))
        plt.show()


if __name__ == "__main__":
    tests_config = {
        "No obstacles": {
            "policies": ["example_policies/example_no_obs/models/model-00030"],
            "env_params": {'obstacles': False}},
        "With obstacles": {
            "policies": ["example_policies/example_with_obs/models/model-00032"],
            "env_params": {'obstacles': True}}}
    
    plotter = TestPlotter(loadpath="example_policies")
    # plotter.run_tests(1000)
    plotter.plot_test_results()
