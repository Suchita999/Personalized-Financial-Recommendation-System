"""
Rules-RAG System for Financial Recommendations
Combines rule-based logic with RAG (Retrieval-Augmented Generation) for personalized financial advice.
"""

from .rule_engine import apply_rules
from .ml_pipeline import load_artifacts, predict_user_profile, match_funds
from .rag_pipeline import build_documents, init_rag, ask
from .fund_matching_rag import load_artifacts as load_fund_artifacts

__all__ = [
    'apply_rules',
    'load_artifacts', 
    'predict_user_profile',
    'match_funds',
    'build_documents',
    'init_rag', 
    'ask',
    'load_fund_artifacts'
]
