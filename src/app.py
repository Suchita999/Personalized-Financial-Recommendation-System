import streamlit as st

def main():
    # Get page from URL parameters
    query_params = st.query_params
    page = query_params.get('page', 'landing')
    
    if page == 'chatbot':
        # Import and run chatbot
        import streamlit_chatbot
        chatbot = streamlit_chatbot.LiteFinancialChatbot()
        chatbot.run()
    else:
        # Default to landing page
        import landing_page
        landing_page.main()

if __name__ == "__main__":
    main()
