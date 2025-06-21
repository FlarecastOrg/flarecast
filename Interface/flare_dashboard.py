import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("🔥 Flare Power Monitor")

uploaded_file = st.file_uploader("Simulated_Ground_Sensor_Data", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview:", df.head())

    min_temp = st.slider("Min flame temp (°C)", 500, 1100, 800)
    filtered = df[df["flame_temp_C"] >= min_temp]

    # Fixed line
    st.line_chart(filtered.set_index("timestamp")[["power_MW"]])