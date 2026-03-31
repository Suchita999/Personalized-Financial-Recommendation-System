import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="FinWise — Financial Recommendation System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def main():
    # Get page from URL parameters
    query_params = st.query_params
    page = query_params.get('page', 'landing')

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
    </style>
    """, unsafe_allow_html=True)

    if page == 'chatbot':
        import streamlit_chatbot
        chatbot = streamlit_chatbot.LiteFinancialChatbot()
        chatbot.run()
    elif page == 'dashboard':
        import dashboard
        dashboard.main()
    else:
        import landing_page
        landing_page.main()

if __name__ == "__main__":
    main()