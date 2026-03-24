import streamlit as st

def main():
    # Hide default Streamlit elements and match chatbot theme
    st.markdown("""
    <style>
        .stDeployButton, #MainMenu, footer, .stSidebar {
            visibility: hidden;
        }
        .stApp {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        }
        body {
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        .main .block-container {
            padding-top: 0;
            padding-bottom: 0;
            max-width: 1200px;
            display: none;
        }
        /* Header styling - same as chatbot but lower position */
        .main-header {
            text-align: center;
            background: linear-gradient(135deg, #a8d5a8 0%, #7fb069 100%);
            color: white;
            padding: 1rem;
            margin: 0;
            position: fixed;
            top: 20px;
            left: 0;
            right: 0;
            z-index: 999;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        /* Force button styling */
        .stButton > button {
            background: #FF5722 !important;
            color: white !important;
            border: none !important;
            padding: 1.5rem 3rem !important;
            border-radius: 50px !important;
            font-size: 1.25rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            box-shadow: 0 8px 25px rgba(255, 87, 34, 0.3) !important;
            margin: 12rem 0 4rem 0 !important;
            transition: all 0.3s ease !important;
        }
        .stButton > button:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 12px 35px rgba(255, 87, 34, 0.4) !important;
            background: #E64A19 !important;
        }
        .stButton > button:active {
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 20px rgba(255, 87, 34, 0.3) !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header - same as chatbot structure but lower position
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 3rem;">Financial Recommendation System</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.5rem;">Your Personal AI-Powered Financial Assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quote container - outside main container, same position as tips on chatbot page
    st.markdown("""
    <div style="max-width: 800px; margin: 8rem 0 1rem 0; padding: 0.5rem; color: black; background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); border-radius: 25px; box-shadow: 0 15px 35px rgba(0,0,0,0.1); border: 2px solid #e9ecef; border-left: 4px solid #4CAF50; position: relative;">
        <div style="position: absolute; top: -10px; left: 20px; background: #4CAF50; color: white; padding: 0.5rem 1rem; border-radius: 15px; font-size: 0.9rem; font-weight: 600;">
            WISDOM
        </div>
        <div style="font-size: 1.8rem; font-weight: 300; color: #4a6741; line-height: 1.4; margin-bottom: 1.5rem; font-style: italic; margin-top: 1rem;">
            "Do not save what is left after spending, but spend what is left after saving"
        </div>
        <div style="font-size: 1.2rem; color: #4CAF50; font-weight: 600; text-align: right; margin-top: 1rem;">
            — Warren Buffett
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Features section - outside main container
    st.markdown("""
    <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 2rem; flex-wrap: wrap;">
        <div style="background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); min-width: 150px; border-top: 3px solid #a8d5a8;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">AI-Powered</div>
            <div style="font-weight: 600; color: #4a6741; margin-bottom: 0.5rem;">Smart Analysis</div>
            <div style="color: #6c757d; font-size: 0.9rem;">Financial insights</div>
        </div>
        <div style="background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); min-width: 150px; border-top: 3px solid #7fb069;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">Real Data</div>
            <div style="font-weight: 600; color: #4a6741; margin-bottom: 0.5rem;">ETF & MF Insights</div>
            <div style="color: #6c757d; font-size: 0.9rem;">Market analysis</div>
        </div>
        <div style="background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); min-width: 150px; border-top: 3px solid #4a6741;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">Personalized</div>
            <div style="font-weight: 600; color: #4a6741; margin-bottom: 0.5rem;">Tailored Advice</div>
            <div style="color: #6c757d; font-size: 0.9rem;">Custom recommendations</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Add spacing before button
    st.markdown('<div style="margin-top: 3rem; display: flex; justify-content: center; align-items: center;">', unsafe_allow_html=True)
    
    # Enhanced button - outside main container with inline styling
    button_html = """
    <button onclick="window.location.href='?page=chatbot'" 
            style="background: #4CAF50; color: white; border: none; padding: 1.5rem 3rem; 
                   border-radius: 50px; font-size: 1.25rem; font-weight: 600; 
                   text-transform: uppercase; letter-spacing: 1px; 
                   box-shadow: 0 8px 25px rgba(76, 175, 80, 0.3); 
                   cursor: pointer; transition: all 0.3s ease; margin: 0 0 4rem 0;"
            onmouseover="this.style.background='#45a049'; this.style.transform='translateY(-3px)'; this.style.boxShadow='0 12px 35px rgba(76, 175, 80, 0.4)'"
            onmouseout="this.style.background='#4CAF50'; this.style.transform='translateY(0)'; this.style.boxShadow='0 8px 25px rgba(76, 175, 80, 0.3)'">
        Start Your Journey Today
    </button>
    """
    st.markdown(button_html, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
