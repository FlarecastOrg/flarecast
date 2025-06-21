import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("🔥 Flare Power Monitor rev3")

# Load CSV
df = pd.read_csv("Simulated_Ground_Sensor_Data.csv")  # ✅ This is enough
print(df.head())  # Check if it's loaded correctly
