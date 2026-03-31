"""
FinWise Chatbot — Premium dark editorial interface
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent))

from cluster_mapping import ClusterMapper
from etf_mf_integration import ETFMFIntegration


def inject_chat_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500&display=swap');

        * { box-sizing: border-box; }

        html, body,
        [data-testid="stAppViewContainer"],
        .stApp {
            background: #0a0f0a !important;
            font-family: 'DM Sans', sans-serif;
        }

        .block-container {
            padding: 0 !important;
            max-width: 100% !important;
        }

        /* ── TOPBAR ── */
        .topbar {
            position: fixed;
            top: 0; left: 0; right: 0;
            height: 60px;
            background: rgba(10,15,10,0.95);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid rgba(138,195,90,0.15);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 2rem;
            z-index: 1000;
        }

        .topbar-brand {
            font-family: 'Playfair Display', serif;
            font-size: 1.3rem;
            font-weight: 600;
            color: #f0ede6;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .topbar-dot {
            width: 8px; height: 8px;
            background: #8ac35a;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        .topbar-sub {
            font-family: 'DM Sans', sans-serif;
            font-size: 0.78rem;
            color: rgba(240,237,230,0.35);
            letter-spacing: 0.08em;
        }

        .topbar-nav {
            display: flex;
            gap: 1rem;
            align-items: center;
        }

        .topbar-nav-link {
            font-family: 'DM Sans', sans-serif;
            font-size: 0.85rem;
            font-weight: 500;
            color: rgba(240,237,230,0.6);
            text-decoration: none;
            padding: 0.4rem 0.8rem;
            border-radius: 20px;
            transition: all 0.2s ease;
        }

        .topbar-nav-link:hover {
            color: #f0ede6;
            background: rgba(138,195,90,0.15);
        }

        .topbar-nav-link.active {
            color: #8ac35a;
            background: rgba(138,195,90,0.12);
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.4; transform: scale(0.7); }
        }

        /* ── CHAT LAYOUT ── */
        .chat-layout {
            display: flex;
            gap: 0;
            min-height: 100vh;
            padding-top: 60px;
            padding-bottom: 80px;
        }

        /* ── SIDEBAR PANEL ── */
        .side-panel {
            width: 280px;
            min-width: 280px;
            background: rgba(138,195,90,0.04);
            border-right: 1px solid rgba(138,195,90,0.1);
            padding: 2rem 1.5rem;
            position: sticky;
            top: 60px;
            height: calc(100vh - 60px);
            overflow-y: auto;
        }

        .panel-heading {
            font-family: 'DM Sans', sans-serif;
            font-size: 0.7rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: rgba(138,195,90,0.6);
            margin-bottom: 1.2rem;
        }

        .tip-item {
            display: flex;
            gap: 10px;
            align-items: flex-start;
            margin-bottom: 1rem;
            font-size: 0.85rem;
            color: rgba(240,237,230,0.45);
            line-height: 1.55;
        }

        .tip-icon {
            color: #8ac35a;
            flex-shrink: 0;
            margin-top: 1px;
        }

        .progress-section {
            margin-top: 2rem;
            padding-top: 2rem;
            border-top: 1px solid rgba(138,195,90,0.1);
        }

        .progress-step {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 0.6rem 0;
        }

        .step-circle {
            width: 28px; height: 28px;
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 0.75rem;
            font-weight: 500;
            flex-shrink: 0;
        }

        .step-circle.done {
            background: #8ac35a;
            color: #0a0f0a;
        }

        .step-circle.active {
            background: rgba(138,195,90,0.2);
            border: 1.5px solid #8ac35a;
            color: #8ac35a;
        }

        .step-circle.pending {
            background: rgba(255,255,255,0.05);
            border: 1.5px solid rgba(138,195,90,0.2);
            color: rgba(240,237,230,0.25);
        }

        .step-label {
            font-size: 0.85rem;
        }

        .step-label.done { color: rgba(240,237,230,0.7); }
        .step-label.active { color: #f0ede6; font-weight: 500; }
        .step-label.pending { color: rgba(240,237,230,0.25); }

        /* ── CHAT AREA ── */
        .chat-area {
            flex: 1;
            display: flex;
            flex-direction: column;
            padding: 2rem 2.5rem;
            max-width: 800px;
            margin: 0 auto;
            width: 100%;
        }

        /* ── MESSAGE BUBBLES ── */
        .msg-row {
            display: flex;
            margin-bottom: 1.2rem;
            animation: fadeUp 0.3s ease;
        }

        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .msg-row.bot { justify-content: flex-start; }
        .msg-row.user { justify-content: flex-end; }

        .avatar {
            width: 34px; height: 34px;
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 0.9rem;
            flex-shrink: 0;
        }

        .avatar.bot {
            background: rgba(138,195,90,0.15);
            border: 1px solid rgba(138,195,90,0.3);
            margin-right: 10px;
            margin-top: 2px;
        }

        .avatar.user {
            background: rgba(100,160,220,0.15);
            border: 1px solid rgba(100,160,220,0.3);
            margin-left: 10px;
            margin-top: 2px;
        }

        .bubble {
            max-width: 72%;
            padding: 0.9rem 1.2rem;
            border-radius: 16px;
            line-height: 1.6;
            font-size: 0.92rem;
        }

        .bubble.bot {
            background: rgba(138,195,90,0.08);
            border: 1px solid rgba(138,195,90,0.15);
            color: rgba(240,237,230,0.85);
            border-bottom-left-radius: 4px;
        }

        .bubble.user {
            background: rgba(100,160,220,0.12);
            border: 1px solid rgba(100,160,220,0.2);
            color: rgba(240,237,230,0.85);
            border-bottom-right-radius: 4px;
        }

        .bubble-meta {
            font-size: 0.72rem;
            color: rgba(240,237,230,0.25);
            margin-top: 0.35rem;
        }

        .bot .bubble-meta { text-align: left; }
        .user .bubble-meta { text-align: right; }

        /* ── INPUT FOOTER ── */
        .input-footer {
            position: fixed;
            bottom: 0; left: 0; right: 0;
            background: rgba(10,15,10,0.97);
            backdrop-filter: blur(12px);
            border-top: 1px solid rgba(138,195,90,0.12);
            padding: 0.9rem 2rem;
            z-index: 999;
        }

        /* ── RECOMMENDATION CARD ── */
        .rec-card {
            background: rgba(138,195,90,0.05);
            border: 1px solid rgba(138,195,90,0.15);
            border-radius: 12px;
            padding: 1rem 1.2rem;
            margin: 0.4rem 0;
        }

        .rec-title {
            font-weight: 500;
            color: #8ac35a;
            font-size: 0.88rem;
            margin-bottom: 0.3rem;
        }

        .rec-body {
            font-size: 0.85rem;
            color: rgba(240,237,230,0.6);
        }

        /* ── STREAMLIT BUTTON overrides ── */
        .stButton > button {
            background: #8ac35a !important;
            color: #0a0f0a !important;
            border: none !important;
            padding: 0.65rem 1.8rem !important;
            border-radius: 100px !important;
            font-family: 'DM Sans', sans-serif !important;
            font-size: 0.88rem !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
        }

        .stButton > button:hover {
            background: #a0d870 !important;
            transform: translateY(-1px) !important;
        }

        /* Ghost / secondary button */
        .stButton.ghost > button {
            background: transparent !important;
            color: rgba(240,237,230,0.55) !important;
            border: 1px solid rgba(138,195,90,0.2) !important;
        }

        .stButton.ghost > button:hover {
            border-color: rgba(138,195,90,0.5) !important;
            color: #f0ede6 !important;
        }

        /* Text input */
        .stTextInput > div > div > input {
            background: rgba(255,255,255,0.04) !important;
            border: 1px solid rgba(138,195,90,0.2) !important;
            border-radius: 12px !important;
            color: #ffffff !important;
            font-family: 'DM Sans', sans-serif !important;
            font-size: 0.92rem !important;
            padding: 0.75rem 1rem !important;
        }

        .stTextInput > div > div > input:focus {
            border-color: rgba(138,195,90,0.55) !important;
            box-shadow: 0 0 0 3px rgba(138,195,90,0.08) !important;
            outline: none !important;
        }

        .stTextInput > div > div > input::placeholder {
            color: rgba(240,237,230,0.2) !important;
        }

        /* Form submit button */
        .stFormSubmitButton > button {
            background: #8ac35a !important;
            color: #0a0f0a !important;
            border: none !important;
            padding: 0.65rem 1.8rem !important;
            border-radius: 100px !important;
            font-family: 'DM Sans', sans-serif !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
        }

        .stFormSubmitButton > button:hover {
            background: #a0d870 !important;
        }

        /* Back link */
        .back-link {
            font-size: 0.82rem;
            color: rgba(138,195,90,0.6);
            cursor: pointer;
            text-decoration: none;
        }

        .back-link:hover { color: #8ac35a; }

        /* Analysis card */
        .analysis-card {
            background: rgba(138,195,90,0.06);
            border: 1px solid rgba(138,195,90,0.18);
            border-radius: 16px;
            padding: 1.4rem 1.6rem;
            margin: 0.6rem 0;
        }

        .analysis-row {
            display: flex;
            justify-content: space-between;
            padding: 0.4rem 0;
            border-bottom: 1px solid rgba(138,195,90,0.08);
            font-size: 0.88rem;
        }

        .analysis-row:last-child { border-bottom: none; }

        .analysis-label { color: rgba(240,237,230,0.4); }
        .analysis-value { color: #f0ede6; font-weight: 500; }
        .analysis-value.green { color: #8ac35a; }
        .analysis-value.red { color: #e06060; }
    </style>
    """, unsafe_allow_html=True)


class LiteFinancialChatbot:
    """Premium FinWise chatbot interface"""

    def __init__(self):
        self.mapper = ClusterMapper()
        self.etf_mf_integration = ETFMFIntegration(Path(__file__).parent.parent)

        if 'messages' not in st.session_state:
            st.session_state.messages = []
            self._add_bot_message("Hello! I'm **FinWise**, your personal financial advisor. I'll analyze your situation and match you with funds from real ETF & mutual fund data.")
            self._add_bot_message("Let's start simple. What is your **annual household income**? *(e.g. 75000)*")

        if 'user_data' not in st.session_state:
            st.session_state.user_data = {}

        if 'current_step' not in st.session_state:
            st.session_state.current_step = 'income'

        if 'classification_done' not in st.session_state:
            st.session_state.classification_done = False

    # ── helpers ──────────────────────────────────────────────────────

    def _add_bot_message(self, content):
        st.session_state.messages.append({
            'role': 'bot', 'content': content,
            'timestamp': datetime.now()
        })

    def _add_user_message(self, content):
        st.session_state.messages.append({
            'role': 'user', 'content': content,
            'timestamp': datetime.now()
        })

    def _parse_number(self, text):
        cleaned = text.replace('$', '').replace(',', '').replace(' ', '').strip()
        try:
            return float(cleaned)
        except ValueError:
            return None

    # ── steps ─────────────────────────────────────────────────────────

    def _step_state(self):
        step = st.session_state.current_step
        done = st.session_state.classification_done
        steps = [
            ('income', 'Annual Income'),
            ('family', 'Family Size'),
            ('expenses', 'Annual Expenses'),
            ('complete', 'Analysis'),
        ]
        result = []
        reached = False
        for key, label in steps:
            if done and key == 'complete':
                result.append(('done', label))
            elif key == step and not done:
                result.append(('active', label))
                reached = True
            elif not reached and key != step:
                result.append(('done', label))
            else:
                result.append(('pending', label))
        return result

    def _process_user_input(self, user_input):
        try:
            if st.session_state.current_step == 'income':
                income = self._parse_number(user_input)
                if income is not None and income >= 0:
                    st.session_state.user_data['income'] = income
                    st.session_state.current_step = 'family'
                    self._add_bot_message(f"Got it — **${income:,.0f}** annual income. How many people are in your household?")
                else:
                    self._add_bot_message("Please enter a plain number, e.g. `75000`.")

            elif st.session_state.current_step == 'family':
                fam = self._parse_number(user_input)
                if fam is not None and 1 <= fam <= 20:
                    st.session_state.user_data['family_size'] = int(fam)
                    st.session_state.current_step = 'expenses'
                    self._add_bot_message(f"A household of **{int(fam)}**. And your total annual expenses? *(e.g. 55000)*")
                else:
                    self._add_bot_message("Please enter a number between 1 and 20.")

            elif st.session_state.current_step == 'expenses':
                exp = self._parse_number(user_input)
                if exp is not None and exp >= 0:
                    st.session_state.user_data['expenses'] = exp
                    st.session_state.current_step = 'complete'
                    self._provide_analysis()
                else:
                    self._add_bot_message("Please enter a plain number, e.g. `60000`.")

        except Exception:
            self._add_bot_message("Something went wrong — could you try again with a simple number?")

    def _provide_analysis(self):
        income = st.session_state.user_data['income']
        family_size = st.session_state.user_data['family_size']
        expenses = st.session_state.user_data['expenses']
        savings_rate = (income - expenses) / income if income > 0 else -1

        if income == 0 or income < 5000:
            bracket = "Zero Income Households"
            description = "No income or very low income requiring assistance"
        elif income > 150000 and savings_rate > 0.15:
            bracket = "High Income Savers"
            description = "High income with strong savings capacity"
        else:
            bracket = "Middle Income Families"
            description = "Middle income with steady cash flow"

        st.session_state.user_data.update({
            'income_bracket': bracket,
            'savings_rate': savings_rate,
            'classification_done': True
        })

        sr_label = "🟢 Healthy" if savings_rate >= 0.1 else "🔴 Needs Attention"

        analysis_msg = (
            f"Here's your financial snapshot:\n\n"
            f"**Income:** ${income:,.0f}  |  "
            f"**Expenses:** ${expenses:,.0f}  |  "
            f"**Family:** {family_size}\n"
            f"**Savings Rate:** {savings_rate:.1%} — {sr_label}\n"
            f"**Profile:** {bracket} — *{description}*\n\n"
            f"Let me pull your personalized recommendations…"
        )
        self._add_bot_message(analysis_msg)

        recs = self._generate_simple_recommendations()
        for rec in recs:
            emoji = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}.get(rec['priority'], '📋')
            self._add_bot_message(
                f"{emoji} **{rec['type']}** *(Priority: {rec['priority']})*\n"
                f"**What to do:** {rec['action']}\n"
                f"**Why:** {rec['reason']}\n"
                f"**Detail:** {rec['details']}"
            )

        self._add_bot_message(
            f"**Summary:** {len(recs)} recommendations · "
            f"{len([r for r in recs if r['priority']=='High'])} high-priority · "
            f"Savings status: {'Needs improvement' if savings_rate < 0.1 else 'On track'}\n\n"
            "Ask me anything — emergency fund targets, retirement math, budget breakdowns…"
        )
        st.session_state.classification_done = True

    def _generate_simple_recommendations(self):
        bracket = st.session_state.user_data.get('income_bracket', 'Middle Income Families')
        savings_rate = st.session_state.user_data.get('savings_rate', 0)
        income = st.session_state.user_data.get('income', 0)
        recs = []

        if bracket == "High Income Savers":
            recs = [{'type': 'Tax Planning', 'priority': 'Medium',
                     'action': 'Optimize tax-advantaged accounts',
                     'reason': 'High bracket benefits from strategic deferral',
                     'details': 'Max 401(k), backdoor Roth, HSA contributions'}]
        elif bracket == "Middle Income Families":
            recs = [
                {'type': 'Emergency Fund', 'priority': 'High',
                 'action': 'Build 3–6 month reserve',
                 'reason': 'Essential financial buffer for families',
                 'details': 'High-yield savings account for liquid access'},
                {'type': 'Retirement', 'priority': 'High',
                 'action': 'Start or increase retirement contributions',
                 'reason': 'Compound growth requires time',
                 'details': '401(k) match first, then IRA or Roth IRA'},
            ]
        else:
            recs = [
                {'type': 'Banking Foundation', 'priority': 'High',
                 'action': 'Open a no-fee checking + savings account',
                 'reason': 'Financial foundation before investing',
                 'details': 'Look for FDIC-insured accounts with no minimums'},
                {'type': 'Assistance Programs', 'priority': 'High',
                 'action': 'Apply for government support programs',
                 'reason': 'Income stabilization comes first',
                 'details': 'SNAP, LIHEAP, local community resources'},
            ]

        if income > 25000:
            user_profile = {'total_income': income,
                            'consensus_cluster_name': bracket,
                            'savings_rate': savings_rate}
            real_recs = self.etf_mf_integration.get_investment_recommendations(user_profile)
            if real_recs:
                rec = real_recs[0]
                recs.append({
                    'type': 'Investment',
                    'priority': 'High',
                    'action': f"Consider {rec.get('Name', 'Selected Fund')}",
                    'reason': rec.get('Category', 'Matched to your risk profile'),
                    'details': f"{rec.get('Category', 'Investment')} — Min: ${rec.get('Min Investment', 1000):,.0f}"
                })

        if savings_rate < 0.1 and income > 0:
            recs.insert(0, {
                'type': 'Savings Rate',
                'priority': 'High',
                'action': 'Increase savings rate to at least 10%',
                'reason': f'Current rate {savings_rate:.1%} is below healthy threshold',
                'details': 'Audit subscriptions, negotiate bills, redirect the delta to savings'
            })

        return recs

    def _handle_follow_up(self, user_input):
        txt = user_input.lower()
        income = st.session_state.user_data.get('income', 0)
        expenses = st.session_state.user_data.get('expenses', 0)
        savings_rate = st.session_state.user_data.get('savings_rate', 0)

        if 'emergency' in txt:
            self._add_bot_message(
                f"For your expense level, a 3–6 month emergency fund is **${expenses*0.25:,.0f}–${expenses*0.5:,.0f}**. "
                "Start with a $1,000 mini-fund and automate monthly transfers."
            )
        elif any(k in txt for k in ['retirement', '401', 'ira', 'roth']):
            self._add_bot_message(
                f"Target 10–15% of income for retirement. That's **${income*0.1:,.0f}–${income*0.15:,.0f}/year** "
                f"(${income*0.1/12:,.0f}–${income*0.15/12:,.0f}/month). Capture any employer match first."
            )
        elif 'save' in txt and 'how much' in txt:
            if savings_rate < 0.1:
                self._add_bot_message(
                    f"Your current rate is {savings_rate:.1%}. Aim for **10%** minimum — "
                    f"that's **${income*0.1/12:,.0f}/month**. Even a 2% increase compounding over 20 years is transformative."
                )
            else:
                self._add_bot_message(
                    f"You're at {savings_rate:.1%} — good! Push toward **15–20%** to accelerate wealth building: "
                    f"${income*0.15/12:,.0f}–${income*0.2/12:,.0f}/month."
                )
        elif 'budget' in txt:
            self._add_bot_message(
                f"Try the **50/30/20 rule**: 50% needs · 30% wants · 20% savings. "
                f"On your income that's **${income*0.2/12:,.0f}/month** earmarked for savings. "
                "Track for 30 days to find leaks."
            )
        elif any(k in txt for k in ['restart', 'again', 'reset', 'new']):
            self._restart_chat()
        else:
            self._add_bot_message(
                "Great question! Based on your profile, I'd focus on the high-priority items above. "
                "Try asking about: *emergency fund*, *retirement*, *how much to save*, or *budget tips*."
            )

    def _restart_chat(self):
        st.session_state.messages = []
        st.session_state.user_data = {}
        st.session_state.current_step = 'income'
        st.session_state.classification_done = False
        self._add_bot_message("Fresh start! What is your **annual household income**?")

    # ── render ────────────────────────────────────────────────────────

    def _render_messages(self):
        for msg in st.session_state.messages:
            ts = msg['timestamp'].strftime("%I:%M %p") if hasattr(msg['timestamp'], 'strftime') else ""
            role = msg['role']
            avatar = "🌿" if role == 'bot' else "👤"
            content = str(msg['content']).replace('\n', '<br>')

            st.markdown(f"""
            <div class="msg-row {role}">
                {'<div class="avatar bot">' + avatar + '</div>' if role == 'bot' else '<div class="avatar user">' + avatar + '</div>' if role == 'user' else ''}
                <div>
                    <div class="bubble {role}">{content}</div>
                    <div class="bubble-meta">{ts}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    def run(self):
        inject_chat_css()

        # ── TOPBAR ──
        st.markdown("""
        <div class="topbar">
            <div class="topbar-brand">
                <span class="topbar-dot"></span>
                FinWise
            </div>
            <div class="topbar-nav">
                <a href="?page=landing" class="topbar-nav-link" target="_blank">Home</a>
                <a href="?page=chatbot" class="topbar-nav-link active">Advisor</a>
                <a href="?page=dashboard" class="topbar-nav-link" target="_blank">Dashboard</a>
            </div>
            <div class="topbar-sub">Financial Recommendation System</div>
        </div>
        """, unsafe_allow_html=True)

        # ── LAYOUT: two columns ──
        col_side, col_main = st.columns([1, 3])

        with col_side:
            st.markdown('<div style="padding-top:70px;">', unsafe_allow_html=True)

            # Dashboard link — show after analysis
            if st.session_state.classification_done:
                st.markdown('<div style="margin-top:.5rem;">', unsafe_allow_html=True)
                if st.button("View Dashboard", key="go_dash"):
                    st.query_params.page = "dashboard"
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

            # Progress steps
            steps = self._step_state()
            numbers = ['1', '2', '3', '✓']
            step_html = '<div class="progress-section"><div class="panel-heading">Your Progress</div>'
            for i, (state, label) in enumerate(steps):
                num = numbers[i] if state != 'done' else '✓'
                step_html += f"""
                <div class="progress-step">
                    <div class="step-circle {state}">{num}</div>
                    <span class="step-label {state}">{label}</span>
                </div>"""
            step_html += '</div>'
            st.markdown(step_html, unsafe_allow_html=True)

            # Tips
            if not st.session_state.classification_done:
                st.markdown("""
                <div style="margin-top:2rem; padding-top:2rem; border-top:1px solid rgba(138,195,90,0.1);">
                    <div class="panel-heading">Tips</div>
                    <div class="tip-item"><span class="tip-icon">›</span> Enter numbers without $ or commas</div>
                    <div class="tip-item"><span class="tip-icon">›</span> Use your household's combined income</div>
                    <div class="tip-item"><span class="tip-icon">›</span> Include rent, food & bills in expenses</div>
                    <div class="tip-item"><span class="tip-icon">›</span> All data stays in your session only</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        with col_main:
            st.markdown('<div style="padding-top:70px; padding-bottom:100px; max-width:740px; margin:0 auto;">', unsafe_allow_html=True)
            self._render_messages()
            st.markdown('</div>', unsafe_allow_html=True)

        # ── FIXED INPUT FOOTER ──
        st.markdown('<div class="input-footer">', unsafe_allow_html=True)

        if st.session_state.classification_done:
            with st.form(key="followup_form", clear_on_submit=True):
                c1, c2, c3 = st.columns([5, 1, 1])
                with c1:
                    user_input = st.text_input("Ask about emergency funds, retirement, budgeting…", placeholder="Ask about emergency funds, retirement, budgeting…",
                                               key="followup_input")
                with c2:
                    send = st.form_submit_button("Send")
                with c3:
                    new_chat = st.form_submit_button("Reset")

                if send and user_input:
                    self._add_user_message(user_input)
                    self._handle_follow_up(user_input)
                    st.rerun()
                elif new_chat:
                    self._restart_chat()
                    st.rerun()
        else:
            with st.form(key="main_form", clear_on_submit=True):
                c1, c2 = st.columns([5, 1])
                with c1:
                    user_input = st.text_input("Type your answer…", placeholder="Type your answer…",
                                               key="main_input")
                with c2:
                    send = st.form_submit_button("Send")

                if send and user_input:
                    self._add_user_message(user_input)
                    self._process_user_input(user_input)
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


def main():
    chatbot = LiteFinancialChatbot()
    chatbot.run()


if __name__ == "__main__":
    main()