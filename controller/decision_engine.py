def decide_action(failure_probability):
    """
    Decides system action based on failure probability.
    """
    if failure_probability < 0.3:
        return "CONTINUE", f"System Healthy (Prob: {failure_probability:.2f})"
    elif failure_probability < 0.7:
        return "WARN", f"Warning: Risk Detected (Prob: {failure_probability:.2f})"
    else:
        return "FALLBACK", f"CRITICAL: SWITCHING TO MANUAL (Prob: {failure_probability:.2f})"

if __name__ == "__main__":
    # Test
    print(decide_action(0.1))
    print(decide_action(0.5))
    print(decide_action(0.9))
