import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def build_model():
    # 1. Load dataset
    print("Loading data...")
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Train a simple model
    print("Training model...")
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    # 3. Print prediction + probability
    print("\n--- Model Output ---")
    sample = X_test[0].reshape(1, -1)
    prediction = model.predict(sample)[0]
    proba = model.predict_proba(sample)[0]
    
    print(f"Sample Input: {sample[0][:5]}...")
    print(f"Prediction: {prediction}")
    print(f"Probabilities: {proba}")
    
    # Save model for later phases
    print("\nSaving model to base_model/model.pkl...")
    joblib.dump(model, "base_model/model.pkl")
    print("Done.")

if __name__ == "__main__":
    build_model()
