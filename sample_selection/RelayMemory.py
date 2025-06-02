import random
from collections import namedtuple, deque
import numpy as np

random.seed(42)
np.random.seed(42)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'state_index', 'next_state_index'))


class ReplayMemory(object):
    """
    Replay Memory implemented as a deque
    """

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)