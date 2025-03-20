import numpy as np

class MetaLearningAgent:
    def __init__(self):
        # Initial weights for LSTM, GRU, and RL signals.
        self.weights = np.array([0.33, 0.33, 0.34], dtype=float)  # Should roughly sum to 1
        self.learning_rate = 0.001  # Adjust as needed

    def predict(self, lstm_pred, gru_pred, rl_decision, technical_factors):
        """
        Combines predictions from LSTM, GRU, and RL models to output a final trading signal.
        
        Parameters:
          - lstm_pred: Float, predicted value from LSTM.
          - gru_pred: Float, predicted value from GRU.
          - rl_decision: Categorical signal ("BUY", "SELL", "HOLD") from the RL ensemble.
          - technical_factors: Dict with additional indicators (must include "last_price").
        
        Returns:
          - final_signal: "BUY", "SELL", or "HOLD".
          - ensemble_confidence: Confidence score (0-100) based on the difference from the last price.
          - ensemble_pred: The ensemble predicted value (for feedback updates).
          - x: The input vector used for prediction.
        """
        # Use a more neutral mapping for the RL output.
        mapping = {"BUY": 0.5, "SELL": -0.5, "HOLD": 0.0}
        rl_val = mapping.get(rl_decision, 0.0)
        
        # Construct the feature vector.
        x = np.array([lstm_pred, gru_pred, rl_val])
        
        # Compute the ensemble predicted price as a weighted sum.
        ensemble_pred = np.dot(self.weights, x)
        
        last_price = technical_factors.get("last_price", 1)
        if ensemble_pred > last_price:
            final_signal = "BUY"
        elif ensemble_pred < last_price:
            final_signal = "SELL"
        else:
            final_signal = "HOLD"
        
        # Calculate a confidence value based on the relative percentage difference.
        ensemble_confidence = min(100, max(0, round(abs(ensemble_pred - last_price) / last_price * 100, 2)))
        
        # Debug output.
        print(f"[META DEBUG] LSTM: {lstm_pred}, GRU: {gru_pred}, RL: {rl_val}, Weights: {self.weights}, "
              f"Ensemble_Pred: {ensemble_pred}, Last Price: {last_price}")
        
        return final_signal, ensemble_confidence, ensemble_pred, x

    def update_weights(self, x, ensemble_pred, actual_price):
        """
        Updates the ensemble weights using a simple gradient descent approach.
        This function should be called with the actual outcome to refine the agent.
        
        Parameters:
          - x: The input vector used in the prediction.
          - ensemble_pred: The ensemble predicted value.
          - actual_price: The actual observed price.
        """
        error = ensemble_pred - actual_price
        grad = 2 * error * x  # Gradient of squared error loss.
        self.weights -= self.learning_rate * grad
        # Normalize weights so that their absolute values sum roughly to 1.
        self.weights = self.weights / np.sum(np.abs(self.weights))
        print(f"[META DEBUG] Updated Weights: {self.weights}")

if __name__ == "__main__":
    # Quick test
    agent = MetaLearningAgent()
    lstm_pred = 1.1050
    gru_pred = 1.1060
    rl_decision = "BUY"
    technical_factors = {"last_price": 1.1040, "RSI": 35, "sentiment": "POSITIVE"}
    signal, confidence, ensemble_pred, x = agent.predict(lstm_pred, gru_pred, rl_decision, technical_factors)
    print(f"Signal: {signal}, Confidence: {confidence}")
