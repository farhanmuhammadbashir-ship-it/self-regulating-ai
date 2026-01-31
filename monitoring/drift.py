import numpy as np
from scipy.stats import ks_2samp

def check_drift(reference_data, new_data):
    """
    Simple drift detection using Kolmogorov-Smirnov test.
    Returns: Drift detected (True/False) and p-value.
    """
    # We'll valid drift if p-value is very low (distributions match poorly)
    statistic, p_value = ks_2samp(reference_data, new_data)
    drift_detected = p_value < 0.05
    return drift_detected, p_value

if __name__ == "__main__":
    # Test
    ref = np.random.normal(0, 1, 1000)
    normal = np.random.normal(0, 1, 1000)
    shifted = np.random.normal(3, 1, 1000) # Big shift

    print(f"Drift (Normal): {check_drift(ref, normal)}")
    print(f"Drift (Shifted): {check_drift(ref, shifted)}")
