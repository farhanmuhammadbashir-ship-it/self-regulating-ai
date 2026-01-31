import numpy as np

class ConfidenceMonitor:
    def __init__(self, window_size=50):
        self.history = []
        self.window_size = window_size

    def add_confidence(self, conf):
        self.history.append(conf)
        if len(self.history) > self.window_size:
            self.history.pop(0)

    def get_trend(self):
        """
        Returns the slope of the confidence trend.
        Negative = Decaying confidence.
        """
        if len(self.history) < 5:
            return 0.0
        
        x = np.arange(len(self.history))
        y = np.array(self.history)
        
        # Linear regression slope (polyfit degree 1)
        slope, _ = np.polyfit(x, y, 1)
        return slope

if __name__ == "__main__":
    monitor = ConfidenceMonitor()
    # Simulate decaying confidence
    for i in range(50):
        monitor.add_confidence(0.9 - (i * 0.01))
    
    print(f"Confidence Trend (Should be negative): {monitor.get_trend()}")
