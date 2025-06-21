import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("🔥 Flare Power Monitor rev4")

# Load CSV
df = pd.read_csv("Simulated_Ground_Sensor_Data.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

# Show preview
st.subheader("Data Preview")
st.write(df.head())

# Show full dataset
st.subheader("Full Dataset")
st.dataframe(df)