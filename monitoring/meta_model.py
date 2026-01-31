import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

class MetaModel:
    def __init__(self):
        self.model = LogisticRegression()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_failure_proba(self, entropy, drift_score, confidence_trend):
        # Input vector: [entropy, drift_score, confidence_trend]
        features = np.array([[entropy, drift_score, confidence_trend]])
        # Return probability of class 1 (Failure)
        return self.model.predict_proba(features)[0][1]

    def save(self, path="monitoring/meta_model.pkl"):
        joblib.dump(self.model, path)
    
    def load(self, path="monitoring/meta_model.pkl"):
        self.model = joblib.load(path)

def train_meta_model():
    """
    Trains the meta-model on synthetic data.
    """
    print("Generating synthetic monitoring data...")
    
    # Normal data (Class 0: No Failure)
    # Low entropy, low drift (using p-value so high p-value means low drift, wait.
    # Drift score in drift.py returned (drift_detected, p_value). 
    # Let's use p-value as the feature? Or maybe a binary 0/1?
    # Walkthrough says "drift_score". 
    # Let's transform drift p-value to a score where High = Drift.
    # Score = 1 - p_value. (If p is 0.0, score is 1.0 (High Drift). If p is 1.0, score is 0.0).
    
    # Normal: Low Entropy (0-0.5), Low Drift Score (0-0.3), Flattish Trend (-0.01 to 0.01)
    n_samples = 1000
    normal_entropy = np.random.uniform(0, 0.5, n_samples)
    normal_drift = np.random.uniform(0, 0.3, n_samples)
    normal_conf = np.random.uniform(-0.01, 0.01, n_samples)
    
    X_normal = np.column_stack((normal_entropy, normal_drift, normal_conf))
    y_normal = np.zeros(n_samples)

    # Failed data (Class 1: Failure)
    # High Entropy, High Drift Score, Negative Trend
    failed_entropy = np.random.uniform(0.6, 1.5, n_samples)
    failed_drift = np.random.uniform(0.6, 1.0, n_samples)
    failed_conf = np.random.uniform(-0.1, -0.02, n_samples) 
    
    X_failed = np.column_stack((failed_entropy, failed_drift, failed_conf))
    y_failed = np.ones(n_samples)
    
    # Combine
    X = np.vstack((X_normal, X_failed))
    y = np.concatenate((y_normal, y_failed))
    
    # Train
    print("Training Meta-Model...")
    meta = MetaModel()
    meta.train(X, y)
    
    # Test
    print("Testing Meta-Model...")
    # Safe scenario
    safe_prob = meta.predict_failure_proba(0.1, 0.1, 0.0)
    print(f"Safe Scenario (Low Ent, Low Drift, Flat Conf) Failure Proba: {safe_prob:.4f}")
    
    # Fail scenario
    fail_prob = meta.predict_failure_proba(0.9, 0.9, -0.05)
    print(f"Fail Scenario (High Ent, High Drift, Dropping Conf) Failure Proba: {fail_prob:.4f}")

    meta.save()
    print("Meta-Model saved to monitoring/meta_model.pkl")

if __name__ == "__main__":
    train_meta_model()
