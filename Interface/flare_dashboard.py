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
flare_sites_df = pd.read_csv("Interface/Updated_Diverse_Fracking_Site_Data_with_Status.csv")

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

# Filter for recommended sites for the top 5 table
recommended_sites_df = flare_sites_df[flare_sites_df['Recommended Site'] == 'Yes'].copy()

# Get top 5 highest scored recommended sites and ensure they are ranked correctly
top_5_sites = recommended_sites_df.nlargest(5, 'Score').copy()
top_5_sites = top_5_sites.sort_values(by='Score', ascending=False).reset_index(drop=True)
top_5_sites['Rank'] = top_5_sites.index + 1

# Initialize session state
if 'selected_company' not in st.session_state:
    st.session_state.selected_company = None
if 'selected_region' not in st.session_state:
    st.session_state.selected_region = None
if 'map_center' not in st.session_state:
    st.session_state.map_center = [flare_sites_df['Latitude'].mean(), flare_sites_df['Longitude'].mean()]
if 'map_zoom' not in st.session_state:
    st.session_state.map_zoom = 5
if 'filter_type' not in st.session_state:
    st.session_state.filter_type = None
if "ai_messages" not in st.session_state:
    st.session_state.ai_messages = []
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = ""

# Create the map
st.subheader("🗺️ Flare Gas Sites Map")

m = folium.Map(
    location=st.session_state.map_center, 
    zoom_start=st.session_state.map_zoom, 
    tiles='OpenStreetMap'
)

# Add markers for each flare site
for index, row in flare_sites_df.iterrows():
    # Determine base color and icon based on site type
    if row['Active Site'] == 'Yes':
        color = 'purple' # Color for active sites
        icon_name = 'industry'
        site_type = 'active'
    elif row['Recommended Site'] == 'Yes':
        if row['Recommendation'] == 'Run AI':
            color = 'blue'
            site_type = 'ai_recommended'
        else:
            color = 'orange'
            site_type = 'btc_recommended'
        icon_name = 'fire'
    else:
        color = 'gray' # Color for non-active, non-recommended sites
        icon_name = 'question-circle'
        site_type = 'other'
    
    # Highlight the selected company
    if (row['Company Name'] == st.session_state.selected_company and 
        st.session_state.selected_region is not None and 
        row['Region'] == st.session_state.selected_region):
        color = 'green'  # Highlight color
        icon_name = 'star'  # Highlight icon
        # Don't skip this site even if it doesn't match current filter
        should_show = True
    else:
        should_show = True
    
    # Skip sites that don't match the current filter (unless it's the selected site)
    if st.session_state.filter_type is not None and not (row['Company Name'] == st.session_state.selected_company and st.session_state.selected_region is not None and row['Region'] == st.session_state.selected_region):
        if st.session_state.filter_type == 'selected':
            if row['Company Name'] != st.session_state.selected_company:
                continue
        elif st.session_state.filter_type == 'active' and row['Active Site'] != 'Yes':
            continue
        elif st.session_state.filter_type == 'ai_recommended' and not (row['Recommended Site'] == 'Yes' and row['Recommendation'] == 'Run AI'):
            continue
        elif st.session_state.filter_type == 'btc_recommended' and not (row['Recommended Site'] == 'Yes' and row['Recommendation'] == 'Mine BTC'):
            continue
        elif st.session_state.filter_type == 'other' and not (row['Active Site'] != 'Yes' and row['Recommended Site'] != 'Yes'):
            continue
    
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

# Map Key (Legend)
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
<style>
    .legend {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 15px;
        background-color: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        display: flex;
        justify-content: space-around;
        align-items: center;
        font-size: 14px;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 8px;
        cursor: pointer;
        padding: 5px 10px;
        border-radius: 3px;
        transition: background-color 0.2s;
    }
    .legend-item:hover {
        background-color: rgba(255, 255, 255, 0.2);
    }
    .legend-icon {
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)

# Create interactive legend buttons
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    if st.button("🏭 Active", key="legend_active", help="Show only active sites"):
        st.session_state.filter_type = "active"
        st.experimental_rerun()

with col2:
    if st.button("🔥 AI", key="legend_ai", help="Show only AI recommended sites"):
        st.session_state.filter_type = "ai_recommended"
        st.experimental_rerun()

with col3:
    if st.button("🔥 BTC", key="legend_btc", help="Show only BTC recommended sites"):
        st.session_state.filter_type = "btc_recommended"
        st.experimental_rerun()

with col4:
    if st.button("⭐ Selected", key="legend_selected", help="Show only selected site"):
        st.session_state.filter_type = "selected"
        st.experimental_rerun()

with col5:
    if st.button("❓ Other", key="legend_other", help="Show only other sites"):
        st.session_state.filter_type = "other"
        st.experimental_rerun()

with col6:
    if st.button("🗑️ Clear", key="clear_filter", help="Show all sites"):
        st.session_state.filter_type = None
        st.experimental_rerun()

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
    if cols[1].button(f"📍 {row['Company Name']}", key=f"btn_table_{index}_{row['Company Name'].replace(' ', '_')}_{row['Region'].replace(' ', '_')}"):
        st.session_state.selected_company = row['Company Name']
        st.session_state.selected_region = row['Region']
        
        # Get coords to re-center the map
        selected_site_coords = flare_sites_df[
            (flare_sites_df['Company Name'] == row['Company Name']) & 
            (flare_sites_df['Region'] == row['Region'])
        ].iloc[0]
        st.session_state.map_center = [selected_site_coords['Latitude'], selected_site_coords['Longitude']]
        st.session_state.map_zoom = 8  # Zoom in on the selected site
        st.experimental_rerun() # Rerun the script to update the map

    cols[2].markdown(f"**${row['Max_Profit']:,.0f}**")
    cols[3].markdown(f"**{row['Lead Time']}**")
    profit_type_emoji = "🤖" if row['Recommendation'] == 'Run AI' else "₿"
    cols[4].markdown(f"{profit_type_emoji} {row['Recommendation']}")
    cols[5].markdown(f"**{row['Score']}**")
    cols[6].markdown(f"**{row['Energy Cost']}**")
    cols[7].markdown(f"**{row['CO₂e Saved']}**")

# Add summary statistics
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Sites", len(flare_sites_df))
with col2:
    st.metric("Total Active Sites", len(flare_sites_df[flare_sites_df['Active Site'] == 'Yes']))
with col3:
    st.metric("Total Recommended Sites", len(flare_sites_df[flare_sites_df['Recommended Site'] == 'Yes']))
with col4:
    total_co2_saved = flare_sites_df['CO₂e Saved'].str.replace(' kg', '', regex=False).astype(float).sum()
    st.metric("Total CO₂e Saved", f"{total_co2_saved:,.0f} kg")

st.markdown("---")  # Add another separator

# AI Chat Interface Integration
def get_fallback_response(prompt: str) -> str:
    """
    Provide fallback responses when API fails with comprehensive FlareCast knowledge
    """
    prompt_lower = prompt.lower()
    
    # Location-based queries
    if any(word in prompt_lower for word in ['california', 'texas', 'permian', 'eagle ford', 'bakken', 'region', 'location', 'state']):
        if 'california' in prompt_lower:
            return "I can help you find sites in California! The dataset includes flare gas sites across different regions. To get specific California site data, please try the AI assistant when it's available, or check the map and table above for sites in your area of interest."
        else:
            return "I can help you find sites by region! The dataset includes sites across multiple regions including Permian Basin, Eagle Ford, Bakken, and others. Check the map above or ask the AI assistant for specific regional data when available."
    
    # Score calculation queries
    elif any(word in prompt_lower for word in ['score', 'calculate', 'calculation', 'how is', 'formula']):
        return "The FlareCast score formula is: Score = Profit / (Deployment_Cost + Energy_Constraint_Penalty + Opportunity_Loss). Profit is calculated differently for BTC mining vs AI inference. BTC Profit = hash_price_btc × site_hashrate × time_window - energy costs. AI Profit = tokens_in_queue × token_value_usd × gpu_utilization - compute costs. Deployment cost includes land cost × lead time + infrastructure setup. Energy constraint penalty penalizes actions requiring more energy than available. Opportunity loss is the difference between best possible profit and chosen action profit."
    
    # Feature-specific queries
    elif any(word in prompt_lower for word in ['energy', 'flare gas', 'power', 'kwh', 'kw']):
        return "FlareCast tracks energy data including energy_available_kWh (instantaneous power from flare gas), energy_variability_score (stability), energy_price_effective (harnessing cost), energy_capacity_kW (max capacity), flare_CO2e_saved_kgph (emissions offset), and flare_regulatory_risk (FLARE Act compliance)."
    
    elif any(word in prompt_lower for word in ['gpu', 'compute', 'hardware', 'utilization']):
        return "Compute data includes gpu_count, gpu_utilization_pct, gpu_temp_C, cpu_utilization_pct, and hardware_ready status. This helps determine if a site can handle AI inference workloads efficiently."
    
    elif any(word in prompt_lower for word in ['bitcoin', 'btc', 'hash', 'mining', 'crypto']):
        return "Bitcoin data includes hash_price_btc ($/TH/s/day), network_difficulty, btc_price_usd, btc_volatility_score, btc_expected_yield_usd, and btc_energy_per_THs (efficiency baseline). These determine mining profitability."
    
    elif any(word in prompt_lower for word in ['inference', 'token', 'llm', 'ai job']):
        return "Inference data includes tokens_in_queue, token_value_usd, token_sla_deadline, avg_token_latency_sec, inference_market_demand, and inference_profit_estimate. This helps evaluate AI workload profitability."
    
    elif any(word in prompt_lower for word in ['battery', 'storage', 'soc', 'charge']):
        return "Battery data includes battery_soc_pct (state of charge), battery_capacity_kWh, battery_efficiency, can_discharge status, and charge_time_remaining_hr. This helps with energy storage optimization."
    
    elif any(word in prompt_lower for word in ['deployment', 'lead time', 'access', 'land cost']):
        return "Deployment constraints include lead_time_days, site_access_score (roads + land type), land_cost_usd_per_day, is_existing_site status, and connection_type (Starlink/fiber/offline)."
    
    elif any(word in prompt_lower for word in ['co2', 'emissions', 'environmental', 'saved']):
        return "Environmental impact is tracked through flare_CO2e_saved_kgph and derived CO2e_saved_total (flare_rate × CO2e_per_mcf × hours_used). This aligns with FLARE Act compliance and emissions reduction goals."
    
    # FlareCast functionality queries
    elif any(word in prompt_lower for word in ['flarecast', 'what is', 'how does', 'platform']):
        return "FlareCast is an AI-powered platform that helps operators decide where to deploy off-grid compute infrastructure—such as Bitcoin mining or LLM inference—at flare gas sites, optimizing for profit, emissions reduction, and regulatory compliance. It compares projected profits from mining versus inference jobs based on real-time data such as hash price, token demand, energy availability, and infrastructure readiness."
    
    # FLARE Act queries
    elif any(word in prompt_lower for word in ['flare act', 'regulatory', 'compliance', 'federal']):
        return "The FLARE Act regulates methane flaring on federal lands. FlareCast incorporates FLARE Act compliance into its site scoring system, helping users choose locations that reduce emissions and comply with federal policies. Environmental benefits are factored into scores based on CO2e emissions saved."
    
    # Hash price queries
    elif any(word in prompt_lower for word in ['hash price', 'hashrate', 'mining']):
        return "Hash price is the expected revenue per terahash per second (TH/s) of mining power. FlareCast uses it to estimate profitability from mining at each site. BTC Profit = hash_price_btc × site_hashrate × time_window - energy costs."
    
    # Lead time queries
    elif any(word in prompt_lower for word in ['lead time', 'deployment', 'setup']):
        return "Lead time is the time required to deploy compute infrastructure at a site. Shorter lead times allow operators to capture profits and environmental benefits sooner, making the site more favorable. Deployment cost includes land_cost_usd_per_day × lead_time_days + infrastructure setup cost."
    
    # Site access queries
    elif any(word in prompt_lower for word in ['site access', 'access score', 'terrain', 'road']):
        return "The site access score estimates how difficult it is to reach and operate at a site. It factors in road access, terrain, weather, and land ownership type. A higher site access score implies more difficulty or cost in deploying hardware, which increases deployment cost and reduces the overall score."
    
    # Energy constraint queries
    elif any(word in prompt_lower for word in ['energy constraint', 'energy penalty', 'insufficient energy']):
        return "Energy constraint penalty = max(0, energy_required - energy_available_kWh) × penalty_weight. It penalizes actions that can't be supported by available energy. If energy is insufficient, FlareCast may recommend charging batteries for future use when workload demand or profitability is higher."
    
    # Opportunity loss queries
    elif any(word in prompt_lower for word in ['opportunity loss', 'opportunity cost']):
        return "Opportunity loss is the difference between the profit of the best available action and the profit of the selected action. It ensures the model prefers actions that maximize total value and discourages suboptimal choices even if they are profitable."
    
    # Site analysis queries
    elif any(word in prompt_lower for word in ['site', 'location', 'flare']):
        return "Based on the flare gas site data, I can help you analyze site viability, profitability, and deployment strategies. FlareCast scores each site based on profitability, lead time, land access, and emissions reduction. The site with the highest FlareCast Score is recommended."
    
    # Bitcoin mining queries
    elif any(word in prompt_lower for word in ['bitcoin', 'btc', 'mining']):
        return "Bitcoin mining at flare gas sites can be highly profitable when energy costs are low and BTC prices are favorable. FlareCast calculates BTC profit as: hash_price_btc × site_hashrate × time_window - (site_hashrate × btc_energy_per_THs × energy_price_effective). The key factors to consider are energy cost, hash price, and deployment lead time."
    
    # AI inference queries
    elif any(word in prompt_lower for word in ['ai', 'inference', 'llm', 'token']):
        return "AI inference at flare gas sites offers stable revenue streams and can be more profitable than Bitcoin mining in certain market conditions. FlareCast calculates inference profit as: tokens_in_queue × token_value_usd × (gpu_utilization_pct / 100) - (gpu_count × energy_price_effective × avg_token_latency_sec / 3600). Consider factors like token demand, energy efficiency, and regulatory compliance."
    
    # Profitability queries
    elif any(word in prompt_lower for word in ['profit', 'revenue', 'earnings']):
        return "Profitability depends on energy costs, market conditions, and site-specific factors. FlareCast uses hash_price_btc, available_hashrate, duration, token_demand, token_value_usd, gpu_utilization_pct, energy_cost, and btc_energy_per_THs to calculate profits from Bitcoin mining and LLM inference jobs."
    
    # Environmental queries
    elif any(word in prompt_lower for word in ['environment', 'co2', 'emissions']):
        return "Flare gas utilization for compute infrastructure can significantly reduce CO₂ emissions by converting waste gas into useful energy. FlareCast calculates CO2e avoided by using flare gas for compute instead of letting it burn or vent, using EPA-based emission factors per MCF of gas captured. This aligns with the FLARE Act and environmental regulations."
    
    # General queries
    else:
        return "I'm here to help with FlareCast's functionality, site analysis, market insights, and deployment strategies. FlareCast is an AI-powered platform that optimizes flare gas site deployment for Bitcoin mining or LLM inference, considering profit, emissions reduction, and regulatory compliance. Feel free to ask about specific sites, profitability analysis, or technical considerations."

def generate_ai_response(prompt: str, flare_sites_df):
    """
    Generate AI response using Gemini API with enhanced FlareCast knowledge and data analysis capabilities
    """
    import requests
    import json
    import os
    
    # Gemini API key (hardcoded from user's provided key)
    gemini_api_key = "AIzaSyCgM7KuCWBULe_ofB9kfby_OpAA4_o8xRY"
    
    # Enhanced data analysis for specific queries
    prompt_lower = prompt.lower()
    
    # Check for location-based queries
    if any(word in prompt_lower for word in ['california', 'texas', 'permian', 'eagle ford', 'bakken', 'region', 'location', 'state']):
        # Filter sites by location
        if 'california' in prompt_lower:
            filtered_sites = flare_sites_df[flare_sites_df['Region'].str.contains('California', case=False, na=False)]
            location_data = f"Found {len(filtered_sites)} sites in California:\n"
            if len(filtered_sites) > 0:
                for idx, site in filtered_sites.head(5).iterrows():
                    location_data += f"- {site['Company Name']}: Score {site['Score']}, BTC Profit ${site['BTC Profit']:,.0f}, AI Profit ${site['AI Profit']:,.0f}\n"
                if len(filtered_sites) > 5:
                    location_data += f"... and {len(filtered_sites) - 5} more sites"
            else:
                location_data = "No sites found in California in the current dataset."
        else:
            # Generic location query
            location_data = f"Available regions in the dataset: {', '.join(flare_sites_df['Region'].unique())}\n"
            location_data += f"Total sites: {len(flare_sites_df)}"
    else:
        location_data = ""
    
    # Check for score calculation queries
    if any(word in prompt_lower for word in ['score', 'calculate', 'calculation', 'how is', 'formula']):
        score_explanation = """
        The FlareCast score formula is: Score = Profit / (Deployment_Cost + Energy_Constraint_Penalty + Opportunity_Loss)
        
        Components:
        1. Profit: Calculated differently for BTC mining vs AI inference
           - BTC Profit = hash_price_btc × site_hashrate × time_window - (site_hashrate × btc_energy_per_THs × energy_price_effective)
           - AI Profit = tokens_in_queue × token_value_usd × (gpu_utilization_pct / 100) - (gpu_count × energy_price_effective × avg_token_latency_sec / 3600)
        
        2. Deployment Cost = land_cost_usd_per_day × lead_time_days + infrastructure setup cost
        
        3. Energy Constraint Penalty = max(0, energy_required - energy_available_kWh) × penalty_weight
        
        4. Opportunity Loss = difference between best possible profit and chosen action profit
        
        The scoring algorithm weighs these factors to determine optimal deployment sites.
        """
    else:
        score_explanation = ""
    
    # Create context-aware prompt with comprehensive FlareCast knowledge
    total_sites = len(flare_sites_df)
    active_sites = len(flare_sites_df[flare_sites_df['Active Site'] == 'Yes'])
    recommended_sites = len(flare_sites_df[flare_sites_df['Recommended Site'] == 'Yes'])
    
    # Get some statistics for context
    avg_score = flare_sites_df['Score'].mean()
    max_score = flare_sites_df['Score'].max()
    min_score = flare_sites_df['Score'].min()
    
    enhanced_prompt = f"""
    You are an AI assistant for FlareCast, an AI-powered platform that helps operators decide where to deploy off-grid compute infrastructure—such as Bitcoin mining or LLM inference—at flare gas sites, optimizing for profit, emissions reduction, and regulatory compliance.

    FLARECAST KNOWLEDGE BASE:
    
    Core Functionality:
    - FlareCast compares projected profits from mining versus inference jobs based on real-time data such as hash price, token demand, energy availability, and infrastructure readiness
    - It selects the most profitable and feasible option for a given site and time
    - Inputs include flare gas energy availability, site deployment cost, GPU and battery states, Bitcoin hash price, LLM token value and demand, land type, lead time, and CO2e emissions offsets
    
    FLARE Act Integration:
    - The FLARE Act regulates methane flaring on federal lands
    - FlareCast incorporates FLARE Act compliance into its site scoring system, helping users choose locations that reduce emissions and comply with federal policies
    - Environmental benefits are factored into scores based on CO2e_saved_kgph × usage_hours
    
    Key Concepts:
    - Hash price: Expected revenue per terahash per second (TH/s) of mining power
    - Lead time: Time required to deploy compute infrastructure at a site (shorter is better)
    - Site access score: Estimates difficulty to reach and operate at a site (road access, terrain, weather, land ownership)
    - Energy constraint penalty: Penalizes actions that can't be supported by available energy
    - Opportunity loss: Difference between best possible profit and chosen action profit
    
    Decision Making:
    - FlareCast evaluates scores for each action (mine BTC, run inference, charge battery) and selects the highest
    - If energy is insufficient, it may recommend charging batteries for future use
    - Can simulate different BTC price or token demand scenarios for future profitability
    - Can explain decisions using natural language justifications
    
    📦 FULL FEATURE LIST FOR FLARECAST AI MODEL:
    
    🟢 1. Generic Site Info
    Feature | Type | Description
    site_id | str | Unique identifier
    timestamp | datetime | Current date/time
    latitude, longitude | float | Geo location
    region | str | (e.g. Texas, ND, etc.)
    is_federal_land | bool | For FLARE Act relevance
    weather_temp_C | float | Ambient temp (for cooling/efficiency)
    
    ⚡ 2. Energy Info (Flare Gas Site)
    Feature | Type | Description
    energy_available_kWh | float | Instantaneous power available from flare gas
    energy_variability_score | float | How spiky or stable the flare gas output is
    energy_price_effective | float | Cost to harness power ($0 if on-site waste gas)
    energy_capacity_kW | float | Max flare capacity (kW)
    flare_CO2e_saved_kgph | float | Estimated emissions offset if used
    flare_regulatory_risk | str | High/Med/Low — based on FLARE Act compliance risk
    
    🧠 3. Compute & GPU Info
    Feature | Type | Description
    gpu_count | int | Number of GPUs on-site
    gpu_utilization_pct | float | Percent currently used
    gpu_temp_C | float | GPU temperature
    cpu_utilization_pct | float | CPU load (if inference uses CPU too)
    hardware_ready | bool | All hardware operational
    
    💰 4. Crypto (Bitcoin) Info
    Feature | Type | Description
    hash_price_btc | float | $/TH/s/day — profitability metric
    network_difficulty | float | Affects mining yield
    btc_price_usd | float | BTC market price
    btc_volatility_score | float | Recent price fluctuations
    btc_expected_yield_usd | float | Profit projection at this hash price
    btc_energy_per_THs | float | Efficiency baseline (e.g. 30 J/TH)
    
    📦 5. Inference Job Info
    Feature | Type | Description
    tokens_in_queue | int | Tokens waiting to be processed
    token_value_usd | float | $/token for processing
    token_sla_deadline | float | Deadline time (hrs) for next job
    avg_token_latency_sec | float | How long it takes to process a token
    inference_market_demand | str | High / Med / Low
    inference_profit_estimate | float | Estimated profit/job ($)
    
    🔋 6. Battery / Storage Info
    Feature | Type | Description
    battery_soc_pct | float | State of charge (%)
    battery_capacity_kWh | float | Total battery size
    battery_efficiency | float | Round-trip energy efficiency (%)
    can_discharge | bool | Whether it's allowed/ready to release power
    charge_time_remaining_hr | float | Time to full
    
    📉 7. Deployment Constraints
    Feature | Type | Description
    lead_time_days | int | Days to set up infra at this site
    site_access_score | float | Composite of roads + land type
    land_cost_usd_per_day | float | Leasing cost
    is_existing_site | bool | True if already deployed site
    connection_type | str | Starlink / fiber / offline
    
    🧾 8. Derived (Optional)
    Feature | Formula Example
    CO2e_saved_total | flare_rate × CO2e_per_mcf × hours_used
    deployment_score | Based on road, land type, infra setup
    profit_btc | hash_price × site_hashrate - cost
    profit_infer | tokens × $/token - GPU_energy_cost
    
    Current Data Summary:
    - Total flare gas sites: {total_sites}
    - Active sites: {active_sites}
    - Recommended sites: {recommended_sites}
    - Average score: {avg_score:.1f}
    - Score range: {min_score:.1f} - {max_score:.1f}
    
    {location_data}
    
    {score_explanation}
    
    User Question: {prompt}
    
    Please provide a helpful, informative response about FlareCast's functionality, site analysis, 
    Bitcoin mining vs AI inference decisions, or any other related topics. Be concise but thorough.
    
    If the user is asking about specific data (like sites in a location or score calculations), 
    provide detailed information based on the data provided above.
    
    Use your knowledge of FlareCast's specific algorithms, formulas, and decision-making processes to provide accurate and detailed responses.
    """
    
    try:
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": enhanced_prompt
                        }
                    ]
                }
            ]
        }
        
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_api_key}",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                content = result['candidates'][0].get('content', {})
                parts = content.get('parts', [])
                if parts and len(parts) > 0:
                    ai_response = parts[0].get('text', '')
                    if ai_response:
                        return ai_response
                    else:
                        return get_fallback_response(prompt)
                else:
                    return get_fallback_response(prompt)
            else:
                return get_fallback_response(prompt)
        else:
            return f"⚠️ **API Error**: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"⚠️ **Error**: {str(e)}"

def generate_site_insights(company_name: str, region: str, flare_sites_df):
    """
    Generate actionable insights for a selected company/site
    """
    # Find the selected site data
    selected_site = flare_sites_df[
        (flare_sites_df['Company Name'] == company_name) & 
        (flare_sites_df['Region'] == region)
    ]
    
    if selected_site.empty:
        return "Site data not found."
    
    site_data = selected_site.iloc[0]
    
    # Create insights prompt
    insights_prompt = f"""
    You are an AI analyst for FlareCast, providing actionable insights for flare gas site deployment decisions.
    
    SITE ANALYSIS REQUEST:
    Generate detailed, actionable insights for the following flare gas site:
    
    Company: {site_data['Company Name']}
    Region: {site_data['Region']}
    Score: {site_data['Score']}
    Recommendation: {site_data['Recommendation']}
    BTC Profit: ${site_data['BTC Profit']:,.0f}
    AI Profit: ${site_data['AI Profit']:,.0f}
    Lead Time: {site_data['Lead Time']}
    Energy Cost: {site_data['Energy Cost']}
    CO₂e Saved: {site_data['CO₂e Saved']}
    Active Site: {site_data['Active Site']}
    
    Please provide a comprehensive analysis with the following structure:
    
    🎯 **EXECUTIVE SUMMARY** (2-3 sentences)
    📊 **PROFITABILITY ANALYSIS** (BTC vs AI comparison)
    ⚡ **ENERGY OPTIMIZATION** (flare gas utilization insights)
    🚀 **DEPLOYMENT STRATEGY** (timeline and resource requirements)
    ⚠️ **RISK ASSESSMENT** (key challenges and mitigation)
    💡 **ACTION PLAN** (specific next steps with timelines)
    
    Make the insights practical and actionable for operators making deployment decisions.
    Use emojis and formatting to make it visually appealing and easy to scan.
    """
    
    try:
        import requests
        import json
        
        # Gemini API key
        gemini_api_key = "AIzaSyCgM7KuCWBULe_ofB9kfby_OpAA4_o8xRY"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": insights_prompt
                        }
                    ]
                }
            ]
        }
        
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_api_key}",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                content = result['candidates'][0].get('content', {})
                parts = content.get('parts', [])
                if parts and len(parts) > 0:
                    insights = parts[0].get('text', '')
                    if insights:
                        return insights
        
        # Fallback insights if API fails
        return f"""
        🎯 **EXECUTIVE SUMMARY**
        {site_data['Company Name']} in {site_data['Region']} shows strong potential with a FlareCast score of {site_data['Score']}. 
        The site is recommended for {site_data['Recommendation']} with projected profits of ${max(site_data['BTC Profit'], site_data['AI Profit']):,.0f}.
        
        📊 **PROFITABILITY ANALYSIS**
        • BTC Mining: ${site_data['BTC Profit']:,.0f} profit potential
        • AI Inference: ${site_data['AI Profit']:,.0f} profit potential
        • Recommended: {site_data['Recommendation']} (higher profit option)
        
        ⚡ **ENERGY OPTIMIZATION**
        • Energy Cost: {site_data['Energy Cost']}
        • CO₂e Savings: {site_data['CO₂e Saved']}
        • Deployment can reduce emissions while generating revenue
        
        🚀 **DEPLOYMENT STRATEGY**
        • Lead Time: {site_data['Lead Time']}
        • Site Status: {'Active' if site_data['Active Site'] == 'Yes' else 'Inactive'}
        • Quick deployment recommended to capture market opportunities
        
        ⚠️ **RISK ASSESSMENT**
        • Market volatility may affect profitability
        • Energy availability fluctuations
        • Regulatory compliance requirements
        
        💡 **ACTION PLAN**
        1. **Immediate (1-2 weeks)**: Finalize site assessment and regulatory compliance
        2. **Short-term (1-2 months)**: Begin infrastructure deployment
        3. **Medium-term (3-6 months)**: Scale operations based on performance
        """
        
    except Exception as e:
        return f"⚠️ **Error generating insights**: {str(e)}"

def generate_executive_summary(site_data):
    """Generate executive summary insights"""
    return f"""
    <div class="summary-card">
        <h3>🎯 Executive Summary</h3>
        <div class="summary-content">
            <p><strong>{site_data['Company Name']}</strong> in <strong>{site_data['Region']}</strong> demonstrates exceptional potential for flare gas utilization with a FlareCast score of <strong>{site_data['Score']}</strong>.</p>
            <p>The site is optimized for <strong>{site_data['Recommendation']}</strong> operations, projecting <strong>${max(site_data['BTC Profit'], site_data['AI Profit']):,.0f}</strong> in potential profits while reducing emissions by <strong>{site_data['CO₂e Saved']}</strong>.</p>
            <div class="key-metrics">
                <div class="metric">
                    <span class="metric-value">{site_data['Score']}</span>
                    <span class="metric-label">FlareCast Score</span>
                </div>
                <div class="metric">
                    <span class="metric-value">${max(site_data['BTC Profit'], site_data['AI Profit']):,.0f}</span>
                    <span class="metric-label">Max Profit Potential</span>
                </div>
                <div class="metric">
                    <span class="metric-value">{site_data['Lead Time']}</span>
                    <span class="metric-label">Lead Time</span>
                </div>
            </div>
        </div>
    </div>
    """

def generate_profitability_analysis(site_data):
    """Generate profitability analysis insights"""
    btc_profit = site_data['BTC Profit']
    ai_profit = site_data['AI Profit']
    recommended = site_data['Recommendation']
    
    return f"""
    <div class="profitability-card">
        <h3>📊 Profitability Analysis</h3>
        <div class="profit-comparison">
            <div class="profit-option {'recommended' if recommended == 'Mine BTC' else ''}">
                <h4>₿ Bitcoin Mining</h4>
                <div class="profit-amount">${btc_profit:,.0f}</div>
                <div class="profit-details">
                    <p>• Hash price dependent</p>
                    <p>• Network difficulty impact</p>
                    <p>• Energy efficiency critical</p>
                </div>
            </div>
            <div class="profit-option {'recommended' if recommended == 'Run AI' else ''}">
                <h4>🤖 AI Inference</h4>
                <div class="profit-amount">${ai_profit:,.0f}</div>
                <div class="profit-details">
                    <p>• Token demand driven</p>
                    <p>• Stable revenue stream</p>
                    <p>• GPU utilization key</p>
                </div>
            </div>
        </div>
        <div class="recommendation">
            <strong>🎯 Recommended: {recommended}</strong> - {recommended.split()[1]} operations show higher profit potential
        </div>
    </div>
    """

def generate_energy_optimization(site_data):
    """Generate energy optimization insights"""
    return f"""
    <div class="energy-card">
        <h3>⚡ Energy Optimization</h3>
        <div class="energy-metrics">
            <div class="energy-metric">
                <span class="metric-icon">💰</span>
                <span class="metric-label">Energy Cost</span>
                <span class="metric-value">{site_data['Energy Cost']}</span>
            </div>
            <div class="energy-metric">
                <span class="metric-icon">🌱</span>
                <span class="metric-label">CO₂e Saved</span>
                <span class="metric-value">{site_data['CO₂e Saved']}</span>
            </div>
            <div class="energy-metric">
                <span class="metric-icon">⚡</span>
                <span class="metric-label">Site Status</span>
                <span class="metric-value">{'Active' if site_data['Active Site'] == 'Yes' else 'Inactive'}</span>
            </div>
        </div>
        <div class="energy-insights">
            <h4>Key Insights:</h4>
            <ul>
                <li>Flare gas utilization converts waste to profit</li>
                <li>Environmental impact reduction through emissions capture</li>
                <li>Energy cost optimization through on-site generation</li>
                <li>Regulatory compliance with FLARE Act requirements</li>
            </ul>
        </div>
    </div>
    """

def generate_deployment_strategy(site_data):
    """Generate deployment strategy insights"""
    return f"""
    <div class="deployment-card">
        <h3>🚀 Deployment Strategy</h3>
        <div class="timeline">
            <div class="timeline-phase">
                <div class="phase-header">
                    <span class="phase-icon">📋</span>
                    <span class="phase-title">Phase 1: Assessment</span>
                    <span class="phase-duration">1-2 weeks</span>
                </div>
                <div class="phase-tasks">
                    <p>• Site feasibility analysis</p>
                    <p>• Regulatory compliance review</p>
                    <p>• Infrastructure planning</p>
                </div>
            </div>
            <div class="timeline-phase">
                <div class="phase-header">
                    <span class="phase-icon">🔧</span>
                    <span class="phase-title">Phase 2: Setup</span>
                    <span class="phase-duration">{site_data['Lead Time']}</span>
                </div>
                <div class="phase-tasks">
                    <p>• Hardware deployment</p>
                    <p>• Energy system integration</p>
                    <p>• Network connectivity setup</p>
                </div>
            </div>
            <div class="timeline-phase">
                <div class="phase-header">
                    <span class="phase-icon">📈</span>
                    <span class="phase-title">Phase 3: Scale</span>
                    <span class="phase-duration">3-6 months</span>
                </div>
                <div class="phase-tasks">
                    <p>• Performance optimization</p>
                    <p>• Capacity expansion</p>
                    <p>• Revenue maximization</p>
                </div>
            </div>
        </div>
    </div>
    """

def generate_risk_assessment(site_data):
    """Generate risk assessment insights"""
    return f"""
    <div class="risk-card">
        <h3>⚠️ Risk Assessment</h3>
        <div class="risk-categories">
            <div class="risk-category high">
                <h4>🔴 High Risk</h4>
                <ul>
                    <li>Market volatility affecting profitability</li>
                    <li>Regulatory policy changes</li>
                </ul>
            </div>
            <div class="risk-category medium">
                <h4>🟡 Medium Risk</h4>
                <ul>
                    <li>Energy availability fluctuations</li>
                    <li>Hardware failure or maintenance</li>
                    <li>Network connectivity issues</li>
                </ul>
            </div>
            <div class="risk-category low">
                <h4>🟢 Low Risk</h4>
                <ul>
                    <li>Site access and logistics</li>
                    <li>Weather-related disruptions</li>
                </ul>
            </div>
        </div>
        <div class="mitigation-strategies">
            <h4>🛡️ Mitigation Strategies:</h4>
            <ul>
                <li>Diversify revenue streams (BTC + AI)</li>
                <li>Implement redundant systems</li>
                <li>Regular regulatory monitoring</li>
                <li>Backup power solutions</li>
            </ul>
        </div>
    </div>
    """

def generate_action_plan(site_data):
    """Generate action plan insights"""
    return f"""
    <div class="action-card">
        <h3>💡 Action Plan</h3>
        <div class="action-timeline">
            <div class="action-item immediate">
                <div class="action-header">
                    <span class="action-icon">⚡</span>
                    <span class="action-title">Immediate Actions (1-2 weeks)</span>
                </div>
                <div class="action-tasks">
                    <p>• Finalize site assessment and due diligence</p>
                    <p>• Secure regulatory approvals and permits</p>
                    <p>• Establish vendor relationships</p>
                    <p>• Develop detailed project timeline</p>
                </div>
            </div>
            <div class="action-item short-term">
                <div class="action-header">
                    <span class="action-icon">🚀</span>
                    <span class="action-title">Short-term Goals (1-2 months)</span>
                </div>
                <div class="action-tasks">
                    <p>• Begin infrastructure deployment</p>
                    <p>• Install and configure hardware</p>
                    <p>• Set up monitoring systems</p>
                    <p>• Conduct initial testing</p>
                </div>
            </div>
            <div class="action-item medium-term">
                <div class="action-header">
                    <span class="action-icon">📈</span>
                    <span class="action-title">Medium-term Objectives (3-6 months)</span>
                </div>
                <div class="action-tasks">
                    <p>• Scale operations based on performance</p>
                    <p>• Optimize energy efficiency</p>
                    <p>• Expand capacity as needed</p>
                    <p>• Implement advanced monitoring</p>
                </div>
            </div>
        </div>
        <div class="success-metrics">
            <h4>📊 Success Metrics:</h4>
            <ul>
                <li>Profitability targets: ${max(site_data['BTC Profit'], site_data['AI Profit']):,.0f}</li>
                <li>Environmental impact: {site_data['CO₂e Saved']} emissions reduction</li>
                <li>Operational efficiency: 95%+ uptime</li>
                <li>ROI timeline: 6-12 months</li>
            </ul>
        </div>
    </div>
    """

# Generate and display actionable insights for selected company
if st.session_state.selected_company and st.session_state.selected_region:
    st.subheader("🎯 Site Analysis & Actionable Insights")
    
    # Get site data
    selected_site = flare_sites_df[
        (flare_sites_df['Company Name'] == st.session_state.selected_company) & 
        (flare_sites_df['Region'] == st.session_state.selected_region)
    ]
    
    if not selected_site.empty:
        site_data = selected_site.iloc[0]
        
        # Add custom CSS for enhanced styling
        st.markdown("""
        <style>
        .tab-container {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            border: 1px solid #e9ecef;
        }
        
        .summary-card, .profitability-card, .energy-card, .deployment-card, .risk-card, .action-card {
            background: #ffffff;
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
            border-left: 5px solid #667eea;
            border: 1px solid #e9ecef;
        }
        
        .summary-card h3, .profitability-card h3, .energy-card h3, .deployment-card h3, .risk-card h3, .action-card h3 {
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.4em;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 10px;
        }
        
        .key-metrics {
            display: flex;
            justify-content: space-around;
            margin-top: 20px;
        }
        
        .metric {
            text-align: center;
            padding: 15px;
            background: #667eea;
            border-radius: 10px;
            color: white;
            min-width: 120px;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
        }
        
        .metric-value {
            display: block;
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .metric-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        
        .profit-comparison {
            display: flex;
            gap: 20px;
            margin: 20px 0;
        }
        
        .profit-option {
            flex: 1;
            padding: 20px;
            border-radius: 10px;
            border: 2px solid #e9ecef;
            text-align: center;
            transition: all 0.3s ease;
            background: #ffffff;
        }
        
        .profit-option.recommended {
            border-color: #27ae60;
            background: #27ae60;
            color: white;
            box-shadow: 0 4px 12px rgba(39, 174, 96, 0.3);
        }
        
        .profit-amount {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
            color: #2c3e50;
        }
        
        .profit-option.recommended .profit-amount {
            color: white;
        }
        
        .profit-option h4 {
            color: #2c3e50;
            margin-bottom: 10px;
        }
        
        .profit-option.recommended h4 {
            color: white;
        }
        
        .profit-details p {
            color: #6c757d;
            margin: 5px 0;
        }
        
        .profit-option.recommended .profit-details p {
            color: rgba(255, 255, 255, 0.9);
        }
        
        .energy-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        
        .energy-metric {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            border: 1px solid #e9ecef;
        }
        
        .metric-icon {
            font-size: 1.5em;
        }
        
        .metric-label {
            font-weight: 500;
            color: #495057;
        }
        
        .metric-value {
            font-weight: bold;
            color: #2c3e50;
        }
        
        .timeline {
            margin: 20px 0;
        }
        
        .timeline-phase {
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            border: 1px solid #e9ecef;
        }
        
        .phase-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }
        
        .phase-icon {
            font-size: 1.5em;
        }
        
        .phase-title {
            font-weight: 600;
            color: #2c3e50;
        }
        
        .phase-duration {
            margin-left: auto;
            background: #667eea;
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.9em;
            font-weight: 500;
        }
        
        .phase-tasks p {
            color: #6c757d;
            margin: 5px 0;
        }
        
        .risk-categories {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        
        .risk-category {
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid;
            border: 1px solid #e9ecef;
        }
        
        .risk-category h4 {
            margin-bottom: 10px;
            font-weight: 600;
        }
        
        .risk-category.high {
            background: #fff5f5;
            border-left-color: #e74c3c;
        }
        
        .risk-category.high h4 {
            color: #c53030;
        }
        
        .risk-category.medium {
            background: #fffbf0;
            border-left-color: #f39c12;
        }
        
        .risk-category.medium h4 {
            color: #d69e2e;
        }
        
        .risk-category.low {
            background: #f0fff4;
            border-left-color: #27ae60;
        }
        
        .risk-category.low h4 {
            color: #2f855a;
        }
        
        .risk-category ul {
            margin: 0;
            padding-left: 20px;
        }
        
        .risk-category li {
            color: #4a5568;
            margin: 5px 0;
        }
        
        .action-timeline {
            margin: 20px 0;
        }
        
        .action-item {
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            border: 1px solid #e9ecef;
        }
        
        .action-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }
        
        .action-icon {
            font-size: 1.5em;
        }
        
        .action-title {
            font-weight: 600;
            color: #2c3e50;
        }
        
        .action-tasks p {
            color: #6c757d;
            margin: 5px 0;
        }
        
        .recommendation {
            text-align: center;
            padding: 15px;
            background: #27ae60;
            color: white;
            border-radius: 10px;
            margin-top: 15px;
            font-weight: bold;
            box-shadow: 0 4px 12px rgba(39, 174, 96, 0.3);
        }
        
        .energy-insights, .mitigation-strategies, .success-metrics {
            margin-top: 20px;
        }
        
        .energy-insights h4, .mitigation-strategies h4, .success-metrics h4 {
            color: #2c3e50;
            margin-bottom: 10px;
            font-weight: 600;
        }
        
        .energy-insights ul, .mitigation-strategies ul, .success-metrics ul {
            margin-left: 20px;
        }
        
        .energy-insights li, .mitigation-strategies li, .success-metrics li {
            margin: 5px 0;
            color: #4a5568;
        }
        
        .summary-content p {
            color: #4a5568;
            line-height: 1.6;
            margin-bottom: 10px;
        }
        
        .summary-content strong {
            color: #2c3e50;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎯 Summary", "📊 Profitability", "⚡ Energy", "🚀 Deployment", "⚠️ Risks", "💡 Action Plan"
        ])
        
        with tab1:
            st.markdown(generate_executive_summary(site_data), unsafe_allow_html=True)
            
        with tab2:
            st.markdown(generate_profitability_analysis(site_data), unsafe_allow_html=True)
            
        with tab3:
            st.markdown(generate_energy_optimization(site_data), unsafe_allow_html=True)
            
        with tab4:
            st.markdown(generate_deployment_strategy(site_data), unsafe_allow_html=True)
            
        with tab5:
            st.markdown(generate_risk_assessment(site_data), unsafe_allow_html=True)
            
        with tab6:
            st.markdown(generate_action_plan(site_data), unsafe_allow_html=True)
    
    st.markdown("---")  # Separator before AI assistant

# Simple AI chat integration
st.subheader("🤖 AI Assistant")

# Display chat messages
for message in st.session_state.ai_messages:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**AI Assistant:** {message['content']}")
    st.markdown("---")

# Chat input with Enter key support
prompt = st.text_input("Ask me about flare gas sites, market analysis, or deployment strategies...", placeholder="How is the score calculated?", key="ai_prompt")

# Send button below the text input
send_button = st.button("Send", key="send_ai")

# Handle both Enter key and button click
if prompt and prompt.strip() and prompt != st.session_state.last_prompt:
    # Update last prompt to prevent duplicate processing
    st.session_state.last_prompt = prompt
    
    # Add user message to chat history
    st.session_state.ai_messages.append({"role": "user", "content": prompt})
    
    # Generate AI response
    response = generate_ai_response(prompt, flare_sites_df)
    
    # Add assistant response to chat history
    st.session_state.ai_messages.append({"role": "assistant", "content": response})
    
    # Clear the input and rerun to show the new messages
    st.experimental_rerun()

st.markdown("---")  # Final separator
