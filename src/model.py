import numpy as np

class LinearModel:
    def __init__(self, dim):
        self.w = np.random.randn(dim)

    def predict(self, x):
        return x @ self.w

    def update(self, x, y, lr):
        pred = self.predict(x)
        grad = x.T @ (pred - y) / len(y)
        self.w -= lr * grad
        return np.mean((pred - y) ** 2)
