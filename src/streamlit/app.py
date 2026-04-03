import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="FinWise — Financial Recommendation System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def main():
    # Initialize session state for navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'landing'
    
    # Force current page if explicitly set
    if 'force_page' in st.session_state:
        st.session_state.current_page = st.session_state.force_page
        del st.session_state.force_page
    
    # Get page from URL parameters
    query_params = st.query_params
    url_page = query_params.get('page', None)
    
    # Update current page if URL parameter is set
    if url_page:
        st.session_state.current_page = url_page
    
    page = st.session_state.current_page

    # Global base CSS
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap');

        .stDeployButton, #MainMenu, footer, header { visibility: hidden; }

        html, body, [data-testid="stAppViewContainer"] {
            margin: 0; padding: 0;
            font-family: 'DM Sans', sans-serif;
        }

        [data-testid="stAppViewContainer"] {
            background: #0d1117;
        }

        section[data-testid="stSidebar"] { display: none !important; }

        .block-container {
            padding: 0 !important;
            max-width: 100% !important;
        }
        
        /* Navigation button styling */
        .nav-button {
            background: linear-gradient(135deg, #1a2332 0%, #2d3748 100%);
            border: 1px solid #4a5568;
            border-radius: 12px;
            padding: 2rem;
            margin: 1rem 0;
            text-align: left;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .nav-button:hover {
            background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(74, 85, 104, 0.3);
        }
        
        .nav-button h3 {
            color: #f7fafc;
            margin: 0 0 0.5rem 0;
            font-size: 1.5rem;
        }
        
        .nav-button p {
            color: #cbd5e0;
            margin: 0;
            font-size: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Page content
    if page == 'chatbot':
        try:
            import streamlit_chatbot
            chatbot = streamlit_chatbot.LiteFinancialChatbot()
            chatbot.run()
                
        except Exception as e:
            st.error(f"Error loading chatbot: {e}")
            st.write("Please check the console for details.")
            
    elif page == 'dashboard':
        try:
            import dashboard
            dashboard.main()
                
        except Exception as e:
            st.error(f"Error loading dashboard: {e}")
            st.write("Please check the console for details.")
            
    else:  # landing page (default)
        try:
            import landing_page
            landing_page.main()
                
        except Exception as e:
            st.error(f"Error loading landing page: {e}")
            st.write("Please check the console for details.")

if __name__ == "__main__":
    main()