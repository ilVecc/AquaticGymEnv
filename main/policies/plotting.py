import os
from collections import deque

import numpy as np
from matplotlib import pyplot as plt


class PerformanceUtils(object):
    
    @staticmethod
    def rolling(array, window):
        shape = array.shape[:-1] + (array.shape[-1] - window, window)
        strides = array.strides + (array.strides[-1],)
        return np.mean(np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides), -1)
    
    @staticmethod
    def show_recent(scores, folder=None):
        scores = np.array(scores)
        offset = 200
        episodes_offset = len(scores) - offset
        scores = scores[-offset:]
        
        plt.figure(figsize=(14, 8))
        
        # MAIN PLOTS
        episodes = np.arange(1, len(scores) + 1) + episodes_offset
        plt.plot(episodes, scores,
                 color='#40e0ff', label="score")
        
        window_short = min(10, int(len(scores) * 0.25))
        scores_short = PerformanceUtils.rolling(scores, window=window_short)
        episodes_short = episodes[window_short:]
        plt.plot(episodes_short, scores_short,
                 color='#4078ff', label="score smooth")
        
        window_long = min(10, int(len(scores_short) * 0.25))
        scores_long = PerformanceUtils.rolling(scores_short, window=window_long)
        episodes_long = episodes_short[window_long:]
        plt.plot(episodes_long, scores_long,
                 color='#4018ff', label="score smoother")
        
        # MEANS
        scores_mean = np.full(scores.shape, np.mean(scores))
        mean_plot = plt.plot(episodes, scores_mean,
                             color='r', label="mean")
        plt.yticks(list(plt.yticks()[0]) + [scores_mean[0]])
        plt.gca().get_yticklabels()[-1].set_color(mean_plot[0].get_color())
        
        success_scores = scores[scores >= 0]
        success_rate = success_scores.size / scores.size * 100
        if success_scores.size > 0:
            scores_success_mean = np.full(scores.shape, np.mean(success_scores))
            mean_success_plot = plt.plot(episodes, scores_success_mean,
                                         color='g', label="mean success")
            plt.yticks(list(plt.yticks()[0]) + [scores_success_mean[0]])
            plt.gca().get_yticklabels()[-1].set_color(mean_success_plot[0].get_color())
        
        # LABELING
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.title(
            "Reward per episode (last 200 episodes) [success rate: {:.2f}%]".format(success_rate))
        plt.legend(loc='lower right')
        if folder is not None:
            plt.savefig(os.path.join(folder, "reward_episode_recent.png"))
        plt.show()
    
    @staticmethod
    def show_overall(scores, folder=None):
        scores = np.array(scores)
        episodes = np.arange(1, len(scores) + 1)
        
        plt.figure(figsize=(14, 8))
        
        factor = 0.1
        window_long = int(len(scores) * factor)
        scores_long = PerformanceUtils.rolling(scores, window=window_long)
        episodes_long = episodes[window_long:]
        plt.plot(episodes_long, scores_long,
                 label="rolling window = {:2.0f}% episodes".format(factor * 100))
        
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.title("Overall Reward per episode")
        plt.legend(loc='lower right')
        if folder is not None:
            plt.savefig(os.path.join(folder, "reward_episode_overall.png"))
        plt.show()


class Buffer:
    
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
    
    def add(self, score):
        if self.count < self.buffer_size:
            self.buffer.append(score)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(score)
    
    def clear(self):
        self.buffer.clear()
        self.count = 0


if __name__ == "__main__":
    
    test_path = "test_2"
    
    buf = Buffer(100000)
    with open(os.path.join(test_path, "checkpoint", "scores.txt"), "r") as log:
        score = log.readline()
        while score != '':
            score = float(score.strip())
            buf.add(score)
            score = log.readline()
    scores = np.array(list(buf.buffer))
    buf.clear()
    
    PerformanceUtils.show_recent(scores, test_path)
    PerformanceUtils.show_overall(scores, test_path)
