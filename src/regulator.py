import numpy as np

class SelfRegulator:
    def __init__(self, lr=0.01):
        self.lr = lr

    def adjust(self, error):
        self.lr *= np.exp(-error)
        return self.lr
