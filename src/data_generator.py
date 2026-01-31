import numpy as np

def generate_environment(n=1000):
    x = np.random.randn(n, 5)
    noise = np.random.normal(0, 0.5, n)
    y = x @ np.array([1.5, -2.0, 0.5, 3.0, -1.0]) + noise
    return x, y
