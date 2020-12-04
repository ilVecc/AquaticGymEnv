from collections import deque

import numpy as np

from policies.dqn_policy import NetworkUtils


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
    buf = Buffer(100000)
    with open("test_new_3/checkpoint/scores.txt", "r") as log:
        score = log.readline()
        while score != '':
            score = float(score.strip())
            buf.add(score)
            score = log.readline()
    scores = np.array(list(buf.buffer))
    NetworkUtils.show_network_performance(scores, folder="test_new_3")
    NetworkUtils.show_overall_performance(scores, folder="test_new_3")
    buf.clear()
