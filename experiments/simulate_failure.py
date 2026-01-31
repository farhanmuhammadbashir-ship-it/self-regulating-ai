import numpy as np
import pandas as pd
import joblib
import time
from sklearn.datasets import load_breast_cancer
# Import our modules
# Note: In a real package structure, these would be imports from the package. 
# Here we assume running from root.
import sys
import os
sys.path.append(os.getcwd())

from monitoring.entropy import calculate_entropy
from monitoring.drift import check_drift
from monitoring.confidence import ConfidenceMonitor
from monitoring.meta_model import MetaModel
from controller.decision_engine import decide_action

def simulate():
    print("--- STARTING SELF-REGULATING AI SIMULATION ---")
    
    # 1. Load Resources
    print("Loading models and data...")
    base_model = joblib.load("base_model/model.pkl")
    meta_model = MetaModel()
    meta_model.load("monitoring/meta_model.pkl")
    
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Reference data for drift (first 100 samples)
    reference_data = X[:100]
    
    # Initialize Monitors
    conf_monitor = ConfidenceMonitor(window_size=10)
    
    print("\n--- PHASE 1: NORMAL OPERATION (Samples 0-20) ---")
    # Simulate Data Stream
    # We will iterate through samples.
    # 0-20: Normal
    # 20-50: Gradually adding noise (Drift + Confusion)
    
    for i in range(50):
        # 1. Get Data Sample
        sample = X[i].copy()
        
        # Inject Failure (Drift) after step 20
        if i > 20:
            noise = np.random.normal(0, 3.0 * (i - 20), sample.shape) # Increasing noise
            sample = sample + noise
            if i == 21:
                print("\n--- PHASE 2: INJECTING FAILURE (Data Drift & Noise) ---")
        
        # 2. Base Model Prediction
        # reshape for single prediction
        sample_input = sample.reshape(1, -1)
        proba = base_model.predict_proba(sample_input)[0]
        max_prob = np.max(proba)
        
        # 3. Calculate Signals
        # A. Entropy
        ent = calculate_entropy(proba)
        
        # B. Confidence Trend
        conf_monitor.add_confidence(max_prob)
        conf_trend = conf_monitor.get_trend()
        
        # C. Drift
        # We check drift against reference. In real stream we'd use a window.
        # Here we compare current sample (extended to batch size 1 for verify? No, KS needs samples)
        # Let's compare a recent window of 50 samples vs reference.
        # For simplicity in this step-by-step simulation, we'll check drift of *this specific sample* vs reference?
        # KS test needs 2 distributions. Comparing 1 sample is hard.
        # Let's accumulate a "current window".
        current_window = X[max(0, i-50):i+1].copy() # use real data window for context
        if i > 20: # corrupt the window too if we are effectively in bad state
             # simplistic approximation: just use the noisy sample as a "batch" of 1? No.
             # Let's just create a small batch of "current" simulated data
             batch_noise = np.random.normal(0, 3.0 * (max(0, i - 20)), (50, 30))
             current_window = X[0:50] + batch_noise

        is_drift, drift_p = check_drift(reference_data.flatten(), current_window.flatten())
        # Convert p -> score (Low p = High Drift)
        drift_score = 1 - drift_p
        
        # 4. Meta-Monitoring (Predict Failure)
        failure_prob = meta_model.predict_failure_proba(ent, drift_score, conf_trend)
        
        # 5. Decision Controller
        action, reason = decide_action(failure_prob)
        
        # Output Log properly
        step_status = f"Step {i:02d} | Ent: {ent:.2f} | Drift: {drift_score:.2f} | ConfTrend: {conf_trend:.3f} --> FailProb: {failure_prob:.2f} | ACTION: {action}"
        
        # Highlight interesting events
        if action == "FALLBACK":
            print(f"\033[91m{step_status}\033[0m") # Red
        elif action == "WARN":
            print(f"\033[93m{step_status}\033[0m") # Yellow
        else:
            print(step_status)
            
        time.sleep(0.1) # readable speed

if __name__ == "__main__":
    simulate()
