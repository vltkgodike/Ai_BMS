# rul_predictor.py
import numpy as np

class RULPredictor:
    def __init__(self, failure_threshold=0.8):
        self.failure_threshold = failure_threshold

    def linear_degradation(self, current_soh, current_cycle, degradation_rate=0.0002):
        if current_soh <= self.failure_threshold:
            return 0
        remaining_soh = current_soh - self.failure_threshold
        rul_cycles = int(remaining_soh / degradation_rate)
        return max(0, rul_cycles)

    def exponential_degradation(self, current_soh, current_cycle, decay_factor=0.9995):
        if current_soh <= self.failure_threshold:
            return 0
        if current_soh > 0 and decay_factor > 0 and decay_factor != 1:
            rul_cycles = int(np.log(self.failure_threshold / current_soh) / np.log(decay_factor))
            return max(0, rul_cycles)
        return 0

    def temperature_adjusted(self, current_soh, current_cycle, avg_temperature=25, base_degradation_rate=0.0002):
        if current_soh <= self.failure_threshold:
            return 0
        temp_factor = 1 + (avg_temperature - 25) * 0.02
        adjusted_degradation_rate = base_degradation_rate * temp_factor
        remaining_soh = current_soh - self.failure_threshold
        rul_cycles = int(remaining_soh / adjusted_degradation_rate)
        return max(0, rul_cycles)

    def ensemble_prediction(self, current_soh, current_cycle, avg_temperature=25):
        predictions = {}
        predictions['linear'] = self.linear_degradation(current_soh, current_cycle)
        predictions['exponential'] = self.exponential_degradation(current_soh, current_cycle)
        predictions['temperature'] = self.temperature_adjusted(current_soh, current_cycle, avg_temperature)

        # Weighted average
        weights = [0.4, 0.3, 0.3]
        methods = ['linear', 'exponential', 'temperature']
        ensemble_rul = sum(predictions[method] * weight for method, weight in zip(methods, weights))
        predictions['ensemble'] = int(ensemble_rul)
        return predictions
