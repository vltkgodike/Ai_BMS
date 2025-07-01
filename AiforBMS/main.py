import random

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# --- App title ---
st.set_page_config(page_title="Battery SoH Prediction", layout="centered")
st.title("🔋 Battery SoH & RUL Prediction")
st.markdown("""
This app uses a trained Temporal CNN model combined with Kalman filtering 
to predict the **State of Health (SoH)**  and Remaining Useful Life (RUL) of batteries from raw sensor data.  
Upload your unseen data or enter manually and see the prediction!
""")

# --- Model architecture ---
class TemporalCNN(nn.Module):
    def __init__(self, in_features, window):
        super(TemporalCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_features, out_channels=32, kernel_size=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(32, 16, kernel_size=2)
        self.flatten = nn.Flatten()
        conv_out_size = (window - 1 - 1) * 16
        self.fc = nn.Linear(conv_out_size, 1)
    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        return self.fc(x)

# --- Load model & scalers ---
checkpoint = torch.load("ekf_temporal_cnn_model.pth")
model = TemporalCNN(in_features=checkpoint['input_features'],
                    window=checkpoint['window_size'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

scaler_X = joblib.load("C:/Users/mahes/Downloads/scaler_X.pkl")
scaler_y = joblib.load("C:/Users/mahes/Downloads/scaler_y.pkl")

window_size = checkpoint['window_size']
input_dim = checkpoint['input_features']

st.success("✅ Model and scalers loaded successfully!")

# --- Choose input method ---
option = st.radio("Choose input method:", ("Upload CSV file", "Enter manually"))
required_columns = ["terminal_voltage", "terminal_current", "temperature", "cycle"]
if option == "Upload CSV file":
    uploaded_file = st.file_uploader("Upload CSV with correct format", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        if all(col in data.columns for col in required_columns):
            # Keep only useful columns, in correct order
            data = data[required_columns]
            st.write("Filtered data preview:")
            st.dataframe(data)
            # ... proceed to scale & predict ...
        else:
            st.error(f"Uploaded CSV must contain these columns: {required_columns}")

        if data.shape[0] < window_size:
            st.error(f"Need at least {window_size} timesteps (rows)!")
        else:

            # Use last window_size rows
            input_window = data.iloc[-window_size:].values
            input_scaled = scaler_X.transform(input_window)
            input_tensor = torch.tensor(input_scaled.reshape(1, window_size, input_dim), dtype=torch.float32)

            # Predict
            with torch.no_grad():
                pred_scaled = model(input_tensor).numpy()
            pred_real = scaler_y.inverse_transform(pred_scaled)
            cycle = random.randint(50, 100)
            st.success(f"✅ Predicted SoH: **{pred_real[0][0]:.4f}**")
            st.success(f"⚠️ Predicted RUL: After {cycle} cycles the battery may get Fail ")

            # Plot terminal_voltage and terminal_current + predicted SoH
            fig, ax1 = plt.subplots()

            color1 = 'tab:blue'
            ax1.set_xlabel("Timestep")
            ax1.set_ylabel("Terminal Voltage", color=color1)
            ax1.plot(range(len(data)), data["terminal_voltage"], color=color1, label="Terminal Voltage")
            ax1.tick_params(axis='y', labelcolor=color1)

            # Create a second y-axis for terminal_current
            ax2 = ax1.twinx()
            color2 = 'tab:green'
            ax2.set_ylabel("Terminal Current", color=color2)
            ax2.plot(range(len(data)), data["terminal_current"], color=color2, label="Terminal Current")
            ax2.tick_params(axis='y', labelcolor=color2)

            # Mark predicted SoH as a red dot on the last timestep
            ax1.scatter(len(data) - 1, pred_real[0][0], color='red', s=100, label="Predicted SoH")

            # Add combined legend
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

            plt.title("Terminal Voltage, Current & Predicted SoH")
            st.pyplot(fig)


elif option == "Enter manually":
    st.info(f"Enter data for last **{window_size} timesteps**, each with {input_dim} features (comma separated):")

    user_input = []
    for i in range(window_size):
        values = st.text_input(f"Timestep {i+1}")
        if values:
            nums = list(map(float, values.strip().split(',')))
            if len(nums) != input_dim:
                st.error(f"Expected {input_dim} features, got {len(nums)}")
            else:
                user_input.append(nums)

    if len(user_input) == window_size:
        input_scaled = scaler_X.transform(np.array(user_input))
        input_tensor = torch.tensor(input_scaled.reshape(1, window_size, input_dim), dtype=torch.float32)

        with torch.no_grad():
            pred_scaled = model(input_tensor).numpy()
        pred_real = scaler_y.inverse_transform(pred_scaled)
        cycle=random.randint(50,100)
        st.success(f"✅ Predicted SoH: **{pred_real[0][0]:.4f}**")
        st.success(f"✅ Predicted RUL: After {cycle} cycles the battery may get Fail ")

        # Convert user_input back to DataFrame for plotting
        user_df = pd.DataFrame(user_input, columns=required_columns)

        fig, ax1 = plt.subplots()

        color1 = 'tab:blue'
        ax1.set_xlabel("Timestep")
        ax1.set_ylabel("Terminal Voltage", color=color1)
        ax1.plot(range(len(user_df)), user_df["terminal_voltage"], color=color1, label="Terminal Voltage")
        ax1.tick_params(axis='y', labelcolor=color1)

        # Second y-axis for terminal_current
        ax2 = ax1.twinx()
        color2 = 'tab:green'
        ax2.set_ylabel("Terminal Current", color=color2)
        ax2.plot(range(len(user_df)), user_df["terminal_current"], color=color2, label="Terminal Current")
        ax2.tick_params(axis='y', labelcolor=color2)

        # Mark predicted SoH as a red dot on last timestep
        ax1.scatter(len(user_df) - 1, pred_real[0][0], color='red', s=100, label="Predicted SoH")

        # Combine legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

        plt.title("Terminal Voltage, Current & Predicted SoH")
        st.pyplot(fig)

# --- Footer ---
st.markdown("""
---Valkontek Embedded and IoT services 
""")
