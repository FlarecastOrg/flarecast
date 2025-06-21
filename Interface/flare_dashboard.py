import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static

# Configure page layout for moderate width
st.set_page_config(
    page_title="Flarecast Dashboard",
    page_icon="🔥",
    layout="centered",  # This gives a moderate width increase
    initial_sidebar_state="expanded"
)

st.title("Flarecast: AI-Powered Flare Gas Optimization")

# Add description
st.markdown("""
**FlareCast** is an AI-powered decision engine that helps operators like MARA choose where to deploy off-grid compute infrastructure (Bitcoin mining or LLM inference) at stranded flare gas sites — maximizing profit while minimizing emissions.

We combine:

🌍 **Geospatial site intelligence** (energy availability, land access, lead time)

⚡ **Dynamic market inputs** (BTC hash price, LLM token demand)

🧠 **AI-based workload optimization** (mine vs infer vs charge battery)

🌱 **Regulatory alignment** with the FLARE Act (emissions impact + compliance)
""")

st.markdown("---")  # Add a separator

# Load CSV
df = pd.read_csv("Simulated_Ground_Sensor_Data.csv")

# Create sample coordinates for flare sites (Texas oil fields)
flare_sites = [
    {"name": "Permian Basin Site 1", "lat": 31.9686, "lon": -102.0077, "status": "Active", "temp": 964.9, "flow": 2.65},
    {"name": "Eagle Ford Site 2", "lat": 28.5383, "lon": -97.8612, "status": "Active", "temp": 945.8, "flow": 2.32},
    {"name": "Barnett Shale Site 3", "lat": 32.7767, "lon": -96.7970, "status": "Inactive", "temp": 969.4, "flow": 2.67},
    {"name": "Haynesville Site 4", "lat": 32.5252, "lon": -93.7502, "status": "Active", "temp": 995.7, "flow": 2.77},
    {"name": "Bakken Site 5", "lat": 47.5515, "lon": -101.0020, "status": "Active", "temp": 942.9, "flow": 2.58}
]

# Create the map
st.subheader("🗺️ Flare Gas Sites Map")

# Create a map centered on Texas
m = folium.Map(location=[32.7767, -96.7970], zoom_start=6, tiles='OpenStreetMap')

# Add markers for each flare site
for site in flare_sites:
    # Choose color based on status
    color = 'red' if site['status'] == 'Active' else 'gray'
    
    # Create popup content
    popup_content = f"""
    <b>{site['name']}</b><br>
    Status: {site['status']}<br>
    Temperature: {site['temp']:.1f}°C<br>
    Flow Rate: {site['flow']:.2f} m³/s
    """
    
    folium.Marker(
        location=[site['lat'], site['lon']],
        popup=folium.Popup(popup_content, max_width=300),
        tooltip=site['name'],
        icon=folium.Icon(color=color, icon='fire', prefix='fa')
    ).add_to(m)

# Display the map
folium_static(m, width=700, height=400)

st.markdown("---")  # Add another separator
