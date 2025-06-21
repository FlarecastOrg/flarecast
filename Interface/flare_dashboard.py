import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static
import json

# Configure page layout
st.set_page_config(
    page_title="Flarecast Dashboard",
    page_icon="🔥",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS for a slightly wider layout
st.markdown("""
    <style>
        .main .block-container {
            max-width: 1100px;
        }
    </style>
""", unsafe_allow_html=True)


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

# Load CSV files
df = pd.read_csv("Interface/Simulated_Ground_Sensor_Data.csv")
flare_sites_df = pd.read_csv("Interface/Fracking_Company_Site_Data.csv")

# Clean and convert profit columns to numeric types
for col in ['BTC Profit', 'AI Profit']:
    # Ensure column is string, remove '$', and convert to numeric, coercing errors
    flare_sites_df[col] = pd.to_numeric(
        flare_sites_df[col].astype(str).str.replace('$', '', regex=False), 
        errors='coerce'
    )

# Drop rows where profit data could not be converted, then calculate max profit
flare_sites_df.dropna(subset=['BTC Profit', 'AI Profit'], inplace=True)

# Calculate max profit for each site
flare_sites_df['Max_Profit'] = flare_sites_df[['BTC Profit', 'AI Profit']].max(axis=1)
flare_sites_df['Profit_Type'] = flare_sites_df.apply(
    lambda row: 'Mine BTC' if row['BTC Profit'] > row['AI Profit'] else 'Run AI Inference', axis=1
)

# Get top 5 highest scored sites and ensure they are ranked correctly
top_5_sites = flare_sites_df.nlargest(5, 'Score').copy()
top_5_sites = top_5_sites.sort_values(by='Score', ascending=False).reset_index(drop=True)
top_5_sites['Rank'] = top_5_sites.index + 1

# Initialize session state
if 'selected_company' not in st.session_state:
    st.session_state.selected_company = None
if 'map_center' not in st.session_state:
    st.session_state.map_center = [flare_sites_df['Latitude'].mean(), flare_sites_df['Longitude'].mean()]
if 'map_zoom' not in st.session_state:
    st.session_state.map_zoom = 5

# Create the map
st.subheader("🗺️ Flare Gas Sites Map")

m = folium.Map(
    location=st.session_state.map_center, 
    zoom_start=st.session_state.map_zoom, 
    tiles='OpenStreetMap'
)

# Add markers for each flare site
for index, row in flare_sites_df.iterrows():
    # Determine base color and icon
    if row['Recommendation'] == 'Run AI':
        color = 'blue'
    else:
        color = 'orange'
    
    icon_name = 'fire'
    
    # Highlight the selected company
    if row['Company Name'] == st.session_state.selected_company:
        color = 'green'  # Highlight color
        icon_name = 'star'  # Highlight icon
    
    popup_content = f"""
    <b>{row['Company Name']}</b><br>
    Score: {row['Score']}<br>
    Energy Cost: {row['Energy Cost']}<br>
    Lead Time: {row['Lead Time']}<br>
    BTC Profit: ${row['BTC Profit']:,.0f}<br>
    AI Profit: ${row['AI Profit']:,.0f}<br>
    CO₂e Saved: {row['CO₂e Saved']}<br>
    <b>Recommendation: {row['Recommendation']}</b>
    """
    
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=folium.Popup(popup_content, max_width=300),
        tooltip=f"{row['Company Name']} - {row['Recommendation']}",
        icon=folium.Icon(color=color, icon=icon_name, prefix='fa')
    ).add_to(m)

# Display the map
folium_static(m, width=1050, height=400)

# Add table of top 5 sites
st.subheader("🏆 Top 5 Highest Profit Sites")

# Table Header
cols = st.columns((1, 4, 2, 2, 3, 2, 2, 2))
headers = ["Rank", "Company", "Max Profit", "Lead Time", "Type", "Score", "Energy Cost", "CO₂e Saved"]
for col, header in zip(cols, headers):
    col.markdown(f"**{header}**")

# Display each row with clickable functionality
for index, row in top_5_sites.iterrows():
    cols = st.columns((1, 4, 2, 2, 3, 2, 2, 2))
    
    cols[0].markdown(f"**{row['Rank']}**")
    
    # Button to select and highlight the company
    if cols[1].button(f"📍 {row['Company Name']}", key=f"btn_{index}_{row['Company Name']}"):
        st.session_state.selected_company = row['Company Name']
        
        # Get coords to re-center the map
        selected_site_coords = flare_sites_df[flare_sites_df['Company Name'] == row['Company Name']].iloc[0]
        st.session_state.map_center = [selected_site_coords['Latitude'], selected_site_coords['Longitude']]
        st.session_state.map_zoom = 12  # Zoom in on the selected site
        st.experimental_rerun() # Rerun the script to update the map

    cols[2].markdown(f"**${row['Max_Profit']:,.0f}**")
    cols[3].markdown(f"**{row['Lead Time']}**")
    profit_type_emoji = "🤖" if row['Profit_Type'] == 'Run AI Inference' else "₿"
    cols[4].markdown(f"{profit_type_emoji} {row['Profit_Type']}")
    cols[5].markdown(f"**{row['Score']}**")
    cols[6].markdown(f"**{row['Energy Cost']}**")
    cols[7].markdown(f"**{row['CO₂e Saved']}**")

# Add summary statistics
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Sites", len(flare_sites_df))
with col2:
    st.metric("AI Recommendations", len(flare_sites_df[flare_sites_df['Recommendation'] == 'Run AI']))
with col3:
    st.metric("BTC Recommendations", len(flare_sites_df[flare_sites_df['Recommendation'] == 'Mine BTC']))
with col4:
    total_co2_saved = flare_sites_df['CO₂e Saved'].str.replace(' kg', '', regex=False).astype(float).sum()
    st.metric("Total CO₂e Saved", f"{total_co2_saved:,.0f} kg")

st.markdown("---")  # Add another separator
