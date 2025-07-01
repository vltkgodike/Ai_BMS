# main_app.py
import streamlit as st
import pandas as pd
import numpy as np
import torch
from rul_predictor import RULPredictor
from soh_model import load_model_and_scalers
import matplotlib.pyplot as plt
from visualization import create_enhanced_plot


# --- App setup ---
st.set_page_config(page_title="Battery SoH & RUL Prediction", layout="centered")
st.title("🔋 Battery SoH & RUL Prediction")
st.markdown("""
This app uses a trained Temporal CNN model combined with advanced RUL prediction methods
to predict the **State of Health (SoH)** and **Remaining Useful Life (RUL)** of batteries from raw sensor data.  
Upload your data or enter manually to see comprehensive predictions!
""")

# --- Sidebar settings ---
st.sidebar.title("⚙️ RUL Prediction Settings")
failure_threshold = st.sidebar.slider("Failure Threshold (%)", 50, 90, 60) / 100
degradation_rate = st.sidebar.slider("Degradation Rate", 0.0001, 0.001, 0.0002, format="%.4f")
st.sidebar.markdown("---")
st.sidebar.markdown("**RUL Methods:**")
show_all_methods = st.sidebar.checkbox("Show all prediction methods", value=True)
# --- Load model & scalers ---
try:
    model, scaler_X, scaler_y, window_size, input_dim = load_model_and_scalers(
        "ekf_temporal_cnn_model.pth", "scaler_X.pkl", "scaler_y.pkl")
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# --- Initialize RUL predictor ---
rul_predictor = RULPredictor(failure_threshold=failure_threshold)

# --- Helper functions ---
def predict_rul_comprehensive(predicted_soh, data, current_cycle=None):
    if current_cycle is None:
        current_cycle = data['cycle'].iloc[-1] if 'cycle' in data.columns else 100
    avg_temp = data['temperature'].mean() if 'temperature' in data.columns else 25

    predictions = rul_predictor.ensemble_prediction(predicted_soh, current_cycle, avg_temp)
    pred_values = [predictions['linear'], predictions['exponential'], predictions['temperature']]
    confidence_range = {
        'min': min(pred_values),
        'max': max(pred_values),
        'std': np.std(pred_values)
    }
    return predictions, confidence_range, avg_temp

def display_results(predictions, confidence_range, predicted_soh, avg_temp):
    st.markdown("## 🎯 Prediction Results")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("🎯 Ensemble RUL", f"{predictions['ensemble']} cycles",
                  delta=f"±{confidence_range['std']:.0f}" if confidence_range['std'] > 1 else None)
        st.metric("🌡️ Avg Temperature", f"{avg_temp:.1f}°C")

    with col2:
        st.metric("📊 Current SoH", f"{predicted_soh * 100:.2f}%")
        st.metric("📈 Confidence Range", f"{confidence_range['min']}-{confidence_range['max']} cycles")

    if predicted_soh > 0.8:
        st.success("🟢 Battery Status: Excellent Health")
    elif predicted_soh > 0.7:
        st.info("🟡 Battery Status: Good Health")
    elif predicted_soh > 0.6:
        st.warning("🟠 Battery Status: Fair Health - Monitor Closely")
    else:
        st.error("🔴 Battery Status: Poor Health - Replacement Recommended")

    if show_all_methods:
        st.markdown("### 📋 Detailed RUL Predictions")
        methods_df = pd.DataFrame({
            'Method': ['Linear Degradation', 'Exponential Decay', 'Temperature Adjusted', 'Ensemble'],
            'RUL (cycles)': [predictions['linear'], predictions['exponential'],
                             predictions['temperature'], predictions['ensemble']],
            'Description': [
                'Assumes constant degradation rate',
                'Accelerating degradation over time',
                'Temperature-compensated degradation',
                'Weighted combination of methods'
            ]
        })
        st.dataframe(methods_df, use_container_width=True)
    st.markdown("## 📈 Comprehensive Analysis")
    enhanced_fig = create_enhanced_plot(data, predicted_soh, predictions, failure_threshold)
    st.pyplot(enhanced_fig)

# --- Input ---
required_columns = ["terminal_voltage", "terminal_current", "temperature", "cycle"]
option = st.radio("Choose input method:", ("Upload CSV file", "Enter manually"))

if option == "Upload CSV file":
    uploaded_file = st.file_uploader("Upload CSV with correct format", type="csv")
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            if all(col in data.columns for col in required_columns):
                data = data[required_columns]
                st.write("📊 Data preview:")
                st.dataframe(data, use_container_width=True)

                if data.shape[0] < window_size:
                    st.error(f"❌ Need at least {window_size} timesteps (rows)!")
                else:
                    # Prepare input for model
                    input_window = data.iloc[-window_size:].values
                    input_scaled = scaler_X.transform(input_window)
                    input_tensor = torch.tensor(input_scaled.reshape(1, window_size, input_dim), dtype=torch.float32)

                    # Predict SoH
                    with torch.no_grad():
                        pred_scaled = model(input_tensor).numpy()
                    predicted_soh = scaler_y.inverse_transform(pred_scaled)[0][0]

                    # Predict RUL using comprehensive methods
                    predictions, confidence_range, avg_temp = predict_rul_comprehensive(predicted_soh, data)

                    display_results(predictions, confidence_range, predicted_soh, avg_temp)

            else:
                st.error(f"CSV must have columns: {required_columns}")
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
elif option == "Enter manually":
    st.info(f"📝 Enter data for last **{window_size} timesteps** (format: voltage,current,temperature,cycle):")

    user_input = []
    for i in range(window_size):
        col1, col2 = st.columns([3, 1])
        with col1:
            values = st.text_input(f"Timestep {i + 1}", placeholder="e.g., 3.7,2.1,25,100")
        with col2:
            if st.button("Clear", key=f"clear_{i}"):
                st.rerun()

        if values:
            try:
                nums = list(map(float, values.strip().split(',')))
                if len(nums) != input_dim:
                    st.error(f"❌ Expected {input_dim} values, got {len(nums)}")
                else:
                    user_input.append(nums)
            except ValueError:
                st.error("❌ Please enter valid numbers separated by commas")
    if len(user_input) == window_size:
        input_array = np.array(user_input)
        input_scaled = scaler_X.transform(input_array)
        input_tensor = torch.tensor(input_scaled.reshape(1, window_size, input_dim), dtype=torch.float32)
        with torch.no_grad():
            pred_scaled = model(input_tensor).numpy()
        predicted_soh = scaler_y.inverse_transform(pred_scaled)[0][0]
        df = pd.DataFrame(user_input, columns=required_columns)
        predictions, confidence_range, avg_temp = predict_rul_comprehensive(predicted_soh, df)
        display_results(predictions, confidence_range, predicted_soh, avg_temp)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
<p><strong>Valkontek Embedded IOT Services Private Limited</strong></p>
<p>Advanced Battery Health Monitoring & RUL Prediction System</p>
</div>
""", unsafe_allow_html=True)
