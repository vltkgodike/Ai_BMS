# visualization.py
import matplotlib.pyplot as plt
import numpy as np

def create_enhanced_plot(data, predicted_soh, predictions, failure_threshold):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    # Plot 1: Terminal Voltage and Current
    color1, color2 = 'tab:blue', 'tab:green'
    ax1.plot(range(len(data)), data["terminal_voltage"], color=color1, label="Terminal Voltage")
    ax1_twin = ax1.twinx()
    ax1_twin.plot(range(len(data)), data["terminal_current"], color=color2, label="Terminal Current")
    ax1.scatter(len(data) - 1, predicted_soh, color='red', s=100, label="Predicted SoH", zorder=5)
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Terminal Voltage", color=color1)
    ax1_twin.set_ylabel("Terminal Current", color=color2)
    ax1.set_title("Voltage, Current & SoH")
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')

    # Plot 2: Temperature over time
    if 'temperature' in data.columns:
        ax2.plot(range(len(data)), data["temperature"], 'r-', linewidth=2, label='Temperature')
        ax2.set_xlabel("Timestep")
        ax2.set_ylabel("Temperature (°C)")
        ax2.set_title("Temperature Profile")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Temperature data\nnot available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title("Temperature Profile")

    # Plot 3: RUL Comparison
    methods = ['Linear', 'Exponential', 'Temperature', 'Ensemble']
    rul_values = [predictions['linear'], predictions['exponential'],
                  predictions['temperature'], predictions['ensemble']]
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']

    bars = ax3.bar(methods, rul_values, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel("RUL (cycles)")
    ax3.set_title("RUL Prediction Comparison")
    ax3.grid(True, alpha=0.3, axis='y')

    for bar, value in zip(bars, rul_values):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(rul_values) * 0.01,
                 f'{value}', ha='center', va='bottom', fontweight='bold')

    # Plot 4: SoH Degradation Projection
    current_cycle = data['cycle'].iloc[-1] if 'cycle' in data.columns else len(data)
    future_cycles = np.arange(current_cycle, current_cycle + predictions['ensemble'])
    if len(future_cycles) > 0:
        future_soh = np.linspace(predicted_soh, failure_threshold, len(future_cycles))
        ax4.plot(future_cycles, future_soh * 100, 'r--', linewidth=2, label='Projected SoH')
        ax4.axhline(y=failure_threshold * 100, color='red', linestyle=':', alpha=0.7, label='Failure Threshold')
        ax4.scatter(current_cycle, predicted_soh * 100, color='blue', s=100, label='Current SoH', zorder=5)

    ax4.set_xlabel("Cycle Number")
    ax4.set_ylabel("SoH (%)")
    ax4.set_title("SoH Degradation Projection")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
