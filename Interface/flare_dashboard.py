import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("🔥 Flare Power Monitor rev2")

uploaded_file = st.file_uploader("Simulated_Ground_Sensor_Data", type="csv")