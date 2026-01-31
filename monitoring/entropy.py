import numpy as np
from scipy.stats import entropy

def calculate_entropy(probabilities):
    """
    Computes entropy of the prediction probabilities.
    Higher entropy = Higher uncertainty (confusion).
    """
    return entropy(probabilities, base=2)

if __name__ == "__main__":
    # Test
    probs = [0.5, 0.5]
    print(f"Entropy of [0.5, 0.5] (Max confusion): {calculate_entropy(probs)}")
    probs = [0.99, 0.01]
    print(f"Entropy of [0.99, 0.01] (Certain): {calculate_entropy(probs)}")
