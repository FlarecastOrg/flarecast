# 🚀 Flarecast AI

**Flarecast** is an AI-powered geospatial platform that helps operators decide:
- ✅ Where to deploy flare-gas-powered compute infrastructure
- ✅ Whether to run Bitcoin mining or AI inference at each site
- ✅ How much CO₂e they can offset — aligning with the FLARE Act

Built in under 6 hours for the MARA Hackathon.

---

## 🗺 Features
- Interactive map of real + mock flare gas sites
- Profit simulation: BTC vs AI inference
- Deployment difficulty scoring (lead time, land access, etc.)
- CO₂ emissions avoided calculator
- FLARE Act compliance indicator
- **🤖 AI Assistant** with Google Gemini integration (pre-configured!)

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard
```bash
streamlit run Interface/flare_dashboard.py
```

### 3. Start Chatting
The AI assistant is ready to use with Google's Gemini API! 🎉

---

## 🤖 AI Integration

The dashboard now includes an AI assistant powered by **Google Gemini** that can:

- **Analyze flare gas sites** for viability and profitability
- **Provide market insights** based on current data
- **Recommend deployment strategies** (Bitcoin mining vs AI inference)
- **Calculate environmental impact** and regulatory compliance

### API Options

1. **Google Gemini API** (Default - ✅ Configured)
   - Your API key is already integrated
   - High-quality responses from Google's latest AI model
   - Fast and reliable

2. **Hugging Face Inference API** (Alternative)
   - Free tier available
   - Multiple model options
   - Requires API key setup

3. **Ollama** (Local - Completely Free)
   - Runs locally on your machine
   - No API limits
   - Privacy-focused

### Usage Examples

Ask the AI assistant questions like:
- "Which site has the highest profit potential?"
- "What are the current trends in Bitcoin mining?"
- "How do energy costs affect AI inference decisions?"
- "What factors should I consider for deployment?"

---

## 📦 Requirements

```bash
pip install -r requirements.txt
```

### Dependencies
- `streamlit` - Web dashboard framework
- `pandas` - Data manipulation
- `folium` - Interactive maps
- `requests` - API calls
- `google-generativeai` - Gemini API integration
- `python-dotenv` - Environment variable management

---

## 📚 Documentation

- **[AI Setup Guide](AI_SETUP_GUIDE.md)** - Complete setup instructions
- **[Example Usage](example_usage.py)** - Code examples
- **[LLM Agent Module](Interface/llm_agent.py)** - Core AI functionality

---

## 🔧 Configuration

The AI assistant works out of the box with Gemini! For alternative APIs:

1. **Hugging Face**: Get a free API key and set `HUGGINGFACE_API_KEY`
2. **Ollama**: Install locally and run `ollama serve`
3. **Switch APIs**: Modify the default in `llm_agent.py`

No setup required for Gemini - it's ready to use! 🚀

---

## 🎯 What's New

- **✅ Gemini Integration**: Now using Google's latest AI model by default
- **🚀 Pre-configured**: Your API key is already integrated
- **⚡ Fast Responses**: Gemini provides quick, high-quality responses
- **🔄 Easy Switching**: Can still use Hugging Face or Ollama if needed
- **Interactive AI Chat Interface** - Built into the dashboard
- **Context-Aware Responses** - AI understands your site data
- **Fallback System** - Works even without external APIs
- **Session Memory** - Chat history persists during your session

```bash
pip install -r requirements.txt
