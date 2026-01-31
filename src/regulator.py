import numpy as np

class SelfRegulator:
    def __init__(self, lr=0.01):
        self.lr = lr

    def adjust(self, error):
        # Dampen the error signal to prevent LR from vanishing too quickly
        self.lr *= np.exp(-error * 0.01)
        return self.lr
