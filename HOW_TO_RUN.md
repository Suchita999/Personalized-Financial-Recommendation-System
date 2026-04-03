# How to Run Your Financial Recommendation System

## Quick Start

### Option 1: Use the Startup Script (Recommended)
```bash
./run_project.sh
```

### Option 2: Manual Startup

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2. Set Up Environment
```bash
# Copy your API key file
cp .env.example .env

# Load environment variables
export $(grep -v '^#' .env | xargs)
```

#### 3. Choose Your Component

**🌟 Full Dashboard (All Features):**
```bash
streamlit run src/app.py --server.port 8501
```

**💬 RAG-Enhanced Chatbot:**
```bash
streamlit run src/streamlit_chatbot.py --server.port 8502
```

**🏠 Landing Page:**
```bash
streamlit run src/landing_page.py --server.port 8503
```

**📊 Advanced Dashboard:**
```bash
streamlit run src/dashboard.py --server.port 8504
```

## 🎯 What Each Component Does

### 1. Full Dashboard (`src/app.py`)
- **Complete system with all features**
- Landing page + chatbot + dashboard integration
- Navigation between all components
- **Best for: Full system demonstration**

### 2. RAG Chatbot (`src/streamlit_chatbot.py`)
- **AI-powered financial advisor with RAG**
- Personalized financial guidance
- ETF/Mutual fund recommendations
- **Best for: Interactive financial advice**

### 3. Landing Page (`src/landing_page.py`)
- **Simple entry point**
- System overview and navigation
- **Best for: First-time users**

### 4. Advanced Dashboard (`src/dashboard.py`)
- **Comprehensive analytics**
- Financial visualizations
- Savings projections
- **Best for: Data analysis and planning**

## 🧪 Test Your RAG Integration

Before running the main app, test the RAG system:
```bash
python test_rag_integration.py
```

**Expected Output:**
- ✅ RAG System: PASSED
- ✅ Chatbot Integration: PASSED
- 🎉 All tests passed! RAG integration is ready.

## 🌐 Access Your Application

Once started, open your browser to:
- **Full Dashboard**: http://localhost:8501
- **Chatbot**: http://localhost:8502  
- **Landing Page**: http://localhost:8503
- **Advanced Dashboard**: http://localhost:8504

## 🔧 Troubleshooting

### "Module not found" errors:
```bash
pip install -r requirements.txt
```

### "API key not working":
```bash
# Check your key is loaded
echo $GEMINI_API_KEY

# Test the RAG system
python test_rag_integration.py
```

### "Port already in use":
```bash
# Kill existing streamlit processes
pkill -f streamlit

# Or use a different port
streamlit run src/app.py --server.port 8505
```

### "RAG system not working":
- The system works without API keys (uses knowledge-based responses)
- With Gemini API key: AI-generated responses
- Check `test_rag_integration.py` output for issues

## 🎯 Recommended Workflow

1. **First Time**: Run `./run_project.sh` and choose option 1 (Full Dashboard)
2. **Test RAG**: Try the chatbot and ask financial questions
3. **Explore**: Navigate between different components
4. **Development**: Use individual components for focused work

## 🌟 Features You'll See

### RAG-Enhanced Chatbot
- Ask: "What's the difference between ETF and mutual fund?"
- Ask: "How should I invest for retirement?"
- Get personalized advice based on your income/expenses

### Financial Dashboard
- Income and expense analysis
- Savings rate projections
- Investment recommendations
- Risk tolerance assessment

### Integration
- Seamless navigation between components
- Consistent user experience
- Real-time data processing

**Your system is ready to run! 🚀**
