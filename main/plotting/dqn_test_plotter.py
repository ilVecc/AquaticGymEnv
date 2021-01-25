import sys
from glob import glob
from pathlib import Path

import gym
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from tests.test_dqn import DQNAquaPolicy
from impl.utils import AquaStateNormalizer


class TestPlotter(object):
    
    def __init__(self, load=False):
        self.results = None
        self.n_tests = 0
        if load:
            self.results = pd.read_pickle("plots/test_results.pickle")
            self.n_tests = self.results.groupby("Environment").count().max().max()
    
    def run_tests(self, n_tests):
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
        df.to_pickle(str(Path("plots") / "test_results.pickle"))
    
    def plot_test_results(self, save=True):
        plt.subplots(1, figsize=(10, 10))
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel("Environment", fontsize=16)
        plt.ylabel("Reward", fontsize=16)
        plt.title(f"Aqua test success ({self.n_tests} runs)", fontsize=24)
        
        sns.boxplot(x="Environment", y="Reward", data=self.results)
        sns.swarmplot(x="Environment", y="Reward", data=self.results.groupby("Environment").sample(min(self.n_tests, 100)))
        if save:
            plt.savefig(str(Path("plots") / "test_overall_rewards.png"))
        plt.show()
        
        means = self.results.groupby("Environment")["Success"].mean() * 100
        print()
        print("Success rates:")
        print(means)


if __name__ == "__main__":
    tests_config = {
        "No obstacles": {
            "policies": glob("dqn_no_obs/*/"),
            "env_params": {'obstacles': False}},
        "With obstacles": {
            "policies": glob("dqn_with_obs/*/"),
            "env_params": {'obstacles': True}}}
    
    plotter = TestPlotter(load=False)
    plotter.run_tests(1000)
    plotter.plot_test_results()
