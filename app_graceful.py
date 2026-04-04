#!/usr/bin/env python3
"""
Streamlit Cloud entry point with graceful ChromaDB handling
"""

import sys
import os
from pathlib import Path
import streamlit as st

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Suppress ChromaDB telemetry
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_TELEMETRY'] = 'False'
os.environ['POSTHOG_DISABLE'] = 'True'

# Set up logging to reduce noise
import logging
logging.basicConfig(level=logging.WARNING)

def check_chromadb():
    """Check if ChromaDB is available"""
    try:
        import chromadb
        return True
    except ImportError:
        return False

def main():
    """Main app with graceful fallback"""
    
    # Check ChromaDB availability
    chromadb_available = check_chromadb()
    
    if not chromadb_available:
        st.error("⚠️ ChromaDB is not available on this deployment")
        st.info("The app will run in limited mode without RAG functionality.")
        st.write("Please contact support to enable full RAG features.")
        
        # Show basic app without RAG
        st.title("🌿 FinWise Financial System")
        st.write("Limited mode - RAG features disabled")
        
        # Load basic components that don't require ChromaDB
        try:
            from ml.cluster_mapping import ClusterMapper
            st.success("✅ ML components loaded")
        except Exception as e:
            st.error(f"❌ ML components failed: {e}")
            
        return
    
    # If ChromaDB is available, run full app
    try:
        from streamlit.app import main as app_main
        app_main()
    except Exception as e:
        st.error(f"App failed to start: {str(e)}")
        st.write("Please check the logs for more details.")

if __name__ == "__main__":
    main()
