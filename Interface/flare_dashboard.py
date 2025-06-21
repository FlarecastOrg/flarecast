import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("🔥 Flare Power Monitor rev3")

# Load CSV
df = pd.read_csv("Simulated_Ground_Sensor_Data.csv")

# Show preview
st.subheader("Data Preview")
st.write(df.head())

# Optional: Show full interactive table
st.subheader("Full Dataset")
st.dataframe(df)
