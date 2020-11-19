from collections import deque

import numpy as np

from dqn import DQNUtils


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
    buf = Buffer(400)
    with open("test_improve_5_linear2/checkpoint/scores.txt", "r") as log:
        score = log.readline()
        while score != '':
            score = float(score.strip())
            buf.add(score)
            score = log.readline()
    scores = np.array(list(buf.buffer))
    DQNUtils.show_network_performance(scores)
