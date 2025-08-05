# PrediTechLLM - Wind Turbine Predictive Maintenance Dashboard

PrediTechLLM is a Streamlit-powered AI dashboard for predictive maintenance and power forecasting of wind turbines. It integrates a CNN-LSTM deep learning model with a custom self-attention mechanism to enhance time series prediction accuracy. The project also includes an LLM-style explanation module and interactive features for real-time monitoring and sharing.

---

## 🚀 Features

- 📊 Loads and preprocesses wind turbine sensor data (`WTFault.csv`)
- 🧠 Hybrid CNN-LSTM model with a custom self-attention layer for forecasting
- 📈 Real-time visualizations of turbine metrics
- 🤖 GPT-style natural language fault explanations (via dummy function, easily replaceable with real OpenAI API)
- 🧾 Maintenance notes logging section
- 🔗 QR code generation for turbine status sharing
- 🌐 Ngrok integration for optional public URL sharing

---

## 🔧 Installation

1. **Clone the repository:**

```bash
git clone https://github.com/nigelterera/PrediTechLLM.git
cd PrediTechLLM