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

# Simple AI chat integration
st.subheader("🤖 AI Assistant")

# Initialize chat history and input state
if "ai_messages" not in st.session_state:
    st.session_state.ai_messages = []
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = ""

# Display chat messages
for message in st.session_state.ai_messages:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**AI Assistant:** {message['content']}")
    st.markdown("---")

# Chat input with Enter key support
prompt = st.text_input("Ask me about flare gas sites, market analysis, or deployment strategies...", key="ai_prompt")

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
