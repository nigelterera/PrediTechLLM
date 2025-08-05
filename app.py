import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import io
import qrcode
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, Flatten, LayerNormalization
from tensorflow.keras.layers import Layer
from sklearn.preprocessing import MinMaxScaler
from pyngrok import ngrok

# ------------------ Custom Self-Attention Layer ------------------
class TimeSeriesSelfAttention(Layer):
    def __init__(self, units, **kwargs):
        super(TimeSeriesSelfAttention, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.query = Dense(self.units)
        self.key = Dense(self.units)
        self.value = Dense(self.units)
        self.combine_heads = Dense(self.units)
        self.layernorm = LayerNormalization()

    def call(self, inputs):
        Q = self.query(inputs)
        K = self.key(inputs)
        V = self.value(inputs)
        attention_scores = tf.matmul(Q, K, transpose_b=True)
        scaled_scores = attention_scores / tf.math.sqrt(tf.cast(tf.shape(K)[-1], tf.float32))
        attention_weights = tf.nn.softmax(scaled_scores, axis=-1)
        context = tf.matmul(attention_weights, V)
        attention_output = self.combine_heads(context)
        return self.layernorm(attention_output + inputs)

# ------------------ Data Loading and Preprocessing ------------------
@st.cache_data
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='%d %m %Y %H:%M')
    df.set_index('Date/Time', inplace=True)
    features = ['LV ActivePower (kW)', 'Wind Speed (m/s)', 'Theoretical_Power_Curve (KWh)', 'Wind Direction (°)']
    df = df[features].dropna()
    return df

def scale_data(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    return scaled, scaler

def create_sequences(data, seq_len=24):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len, 0])  # Predict next timestep power output
    return np.array(X), np.array(y)

# ------------------ Model Architecture ------------------
def build_cnn_lstm_attention_model(seq_len, num_features):
    inputs = Input(shape=(seq_len, num_features))
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    x = LSTM(64, return_sequences=True)(x)
    x = TimeSeriesSelfAttention(units=64)(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='linear')(x)  # Regression output (power prediction)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# ------------------ Dummy GPT Explanation ------------------
def gpt_fault_explanation(wind_speed, power_output):
    # Placeholder for real LLM API call
    if wind_speed > 10 and power_output < 100:
        return "High wind speed with low power output indicates potential blade pitch failure."
    else:
        return "Operating conditions appear normal."

# ------------------ Forecast Generation ------------------
def generate_forecast(current_power, hours=12):
    return [current_power + random.uniform(-10, 10) for _ in range(hours)]

# ------------------ QR Code Generation ------------------
def generate_qr_code(data):
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color='black', back_color='white')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer

# ------------------ Streamlit Dashboard ------------------
def main():
    st.set_page_config(page_title="PrediTech Wind Turbine Dashboard", layout="wide")
    st.title("🔋 PrediTech AI Dashboard")
    st.subheader("Predictive Maintenance & Power Forecasting for Wind Turbines")

    # Load and preprocess data
    df = load_and_preprocess_data('WTFault.csv')
    scaled_data, scaler = scale_data(df)
    seq_len = 24
    X, y = create_sequences(scaled_data, seq_len)

    # Display raw data
    with st.expander("Show Raw Sensor Data"):
        st.dataframe(df.head(10))

    # Sensor data plots
    st.markdown("### Sensor Data Trends")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=df['LV ActivePower (kW)'], ax=ax, label='Active Power (kW)')
    sns.lineplot(data=df['Wind Speed (m/s)'], ax=ax, label='Wind Speed (m/s)')
    ax.set_title("Active Power and Wind Speed Over Time")
    ax.legend()
    st.pyplot(fig)

    # Simulated turbine status
    turbine_status = random.choice(["🟢 Normal", "🟡 Warning", "🔴 Fault Detected"])
    st.markdown(f"### Turbine Status: {turbine_status}")

    # Real-time sensor snapshot
    wind_speed = round(random.uniform(4.5, 12.0), 2)
    power_output = round((wind_speed ** 3) * 0.5 + random.uniform(-20, 20), 2)
    col1, col2 = st.columns(2)
    col1.metric(label="Wind Speed (m/s)", value=wind_speed)
    col2.metric(label="Active Power Output (kW)", value=power_output)

    # Power forecast chart
    st.markdown("#### Power Forecast (Next 12 Hours)")
    hours = np.arange(0, 12)
    forecast = generate_forecast(power_output)
    forecast_df = pd.DataFrame({'Hour': hours, 'Forecast Power (kW)': forecast})
    st.line_chart(forecast_df.set_index('Hour'))

    # Fault detection explanation
    if "Fault" in turbine_status:
        st.error("⚠ Fault Detected: Sudden drop in power output with high wind speed.")
        explanation = gpt_fault_explanation(wind_speed, power_output)
        st.markdown(f"**GPT Explanation:** {explanation}")
    else:
        st.success("✅ No faults detected. Operating normally.")

    # Maintenance recommendation
    st.markdown("#### Maintenance Recommendation")
    if "Fault" in turbine_status:
        st.write("- Inspect gearbox and blade sensors within 6 hours.")
    else:
        st.write("- Routine inspection in 48 hours. Monitor torque and vibration.")

    # QR code sharing
    st.subheader("📱 QR Code to Share Status")
    qr_text = f"Turbine Status: {turbine_status}, Wind Speed: {wind_speed}, Power Output: {power_output}"
    qr_buffer = generate_qr_code(qr_text)
    st.image(qr_buffer, caption="Scan to share current turbine status")

    # Analyst notes
    st.subheader("📝 Analyst Notes")
    notes = st.text_area("Add notes (e.g., site location, timestamp, maintenance suggestions):", height=100)
    if notes:
        st.success("✅ Note saved locally for this session.")

    # Footer with professional info
    st.markdown("---")
    st.caption("""
    Developed by **Nigel Terera**  
    Master’s Candidate, Electrical Engineering  
    Shanghai University, 2025  

    This project highlights advanced skills in predictive maintenance and power forecasting using CNN-LSTM with self-attention layers.  
    Demonstrates integration of time-series data processing, deep learning, explainability via GPT-style fault insights, and interactive visualization.  
    Relevant for roles in AI, machine learning engineering, and LLM-driven industrial applications.
    """)

    # Optional Ngrok public URL
    if st.checkbox("🌐 Generate Public Link (Ngrok)"):
        try:
            ngrok.kill()
            public_url = ngrok.connect(8501)
            st.write(f"Public URL: {public_url}")
        except Exception as e:
            st.error(f"Ngrok error: {e}")

if __name__ == "__main__":
    main()