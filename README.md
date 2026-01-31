# Self-Regulating AI System

> **A minimal AI system that detects and reacts to its own failure during deployment without ground truth.**

## 1. The Problem: Silent AI Failure
In production, AI models often fail silently. Distributions shift, confidence drops, and the model starts making bad predictions with high certainty. Without labels (which often arrive days later), we need a way to know **right now** if the AI is broken.

## 2. Architecture

```mermaid
graph TD
    Data[Incoming Data Stream] --> BaseModel[Base Model (RandomForest)]
    Data --> Monitoring[Monitoring Layer]
    
    BaseModel --> Pred[Predictions & Probs]
    Pred --> Monitoring
    
    subgraph "Failure Signals"
    Monitoring --> Entropy[Entropy (Uncertainty)]
    Monitoring --> Drift[Drift (Dist. Shift)]
    Monitoring --> Conf[Confidence Trend]
    end
    
    Entropy --> MetaModel[Meta-Monitoring Model]
    Drift --> MetaModel
    Conf --> MetaModel
    
    MetaModel --> FailProb[Failure Probability]
    FailProb --> Controller[Decision Controller]
    
    Controller --> Action{Action}
    Action --> Continue[CONTINUE (Healthy)]
    Action --> Warn[WARN (Risk)]
    Action --> Fallback[FALLBACK (Manual Override)]
```

## 3. Failure Signals Explained

1.  **Entropy (Uncertainty):** Measures how "confused" the model is. High entropy = flat probability distribution (e.g., [0.5, 0.5]).
2.  **Distribution Shift (Drift):** Measures if the incoming data looks different from training data (using statistical tests like KS-test).
3.  **Confidence Decay:** Tracks if the model's average confidence is trending downward over time.

## 4. Experiment Setup
We simulated a production environment:
- **Phase 1 (Steps 0-20):** Normal operation. Clean breast cancer dataset samples.
- **Phase 2 (Steps 20-50):** We injected increasing noise to simulate sensor failure or major data drift.

**Goal:** The system must detect this *without knowing the true labels* and switch to FALLBACK mode.

## 5. Results
Running `python experiments/simulate_failure.py`:

```text
Step 18 | Ent: 0.15 | Drift: 0.05 | ConfTrend: 0.001 --> FailProb: 0.00 | ACTION: CONTINUE
...
--- PHASE 2: INJECTING FAILURE ---
Step 22 | Ent: 0.85 | Drift: 0.99 | ConfTrend: -0.050 --> FailProb: 0.98 | ACTION: FALLBACK
```

The system successfully detected the drift and high uncertainty, transitioning the decision engine to **FALLBACK** automatically.

## 6. Why This Matters
This architecture proves that we can build **self-governing AI**. Instead of blindly trusting a model, we wrap it in a control system that validates its health in real-time, preventing catastrophic errors in autonomous systems.

## Usage
1. `pip install -r requirements.txt`
2. `python base_model/model.py` (Train base)
3. `python monitoring/meta_model.py` (Train monitor)
4. `python experiments/simulate_failure.py` (Run simulation)

