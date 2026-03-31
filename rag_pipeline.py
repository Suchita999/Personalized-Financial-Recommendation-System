"""
rag_pipeline.py — RAG chatbot: embed documents in ChromaDB, answer with Gemini.

Dependencies:
    pip install langchain langchain-google-genai langchain-community
    pip install chromadb sentence-transformers

Requires:
    GOOGLE_API_KEY env var (free from https://ai.google.dev)
"""

import os
import pandas as pd

_rag_cache = {}


# ─────────────────────────────────────────────
# 1. BUILD DOCUMENT CORPUS
# ─────────────────────────────────────────────

def build_documents(fund_feat, rec_map=None):
    """
    Create text documents from structured data for the vector store.

    Document types:
      - Profile descriptions (6 composite investor personas)
      - Fund descriptions (from Kaggle ETF/MF dataset)
      - Recommendation rationales (why each profile gets certain funds)
      - Financial education (IRS rules, account types, investing basics)

    Returns: list[dict] with keys 'text' and 'metadata'
    """
    docs = []

    # ── 1. Profile descriptions (6 composite personas) ──
    profile_texts = {
        ('aggressive', 'high_saver'): (
            "Investor profile: Aggressive risk tolerance with high savings behaviour. "
            "This investor seeks maximum growth and can handle significant market volatility. "
            "They have strong savings discipline and surplus cash for investing. "
            "Preferred funds: high-risk, high-return equity funds. "
            "Suitable asset allocation: 80-100% stocks, 0-20% bonds."
        ),
        ('aggressive', 'low_saver'): (
            "Investor profile: Aggressive risk tolerance with low savings behaviour. "
            "Growth-oriented but budget-stretched. Needs low-cost, high-growth funds. "
            "Index ETFs are ideal due to low expense ratios. "
            "Suitable asset allocation: 70-90% stocks, 10-30% bonds."
        ),
        ('moderate', 'high_saver'): (
            "Investor profile: Moderate risk tolerance with high savings behaviour. "
            "A balanced investor with room for some risk, good savings discipline. "
            "Prefers a mix of equity and bond funds for steady growth. "
            "Suitable asset allocation: 50-70% stocks, 30-50% bonds."
        ),
        ('moderate', 'low_saver'): (
            "Investor profile: Moderate risk tolerance with low savings behaviour. "
            "Balanced but cost-sensitive. Limited surplus requires affordable funds. "
            "Target-date funds or balanced index funds work well. "
            "Suitable asset allocation: 40-60% stocks, 40-60% bonds."
        ),
        ('conservative', 'high_saver'): (
            "Investor profile: Conservative risk tolerance with high savings behaviour. "
            "Prioritises capital preservation and stable returns. Strong cash position. "
            "Bond funds, treasury funds, and high-quality dividend funds are suitable. "
            "Suitable asset allocation: 20-40% stocks, 60-80% bonds."
        ),
        ('conservative', 'low_saver'): (
            "Investor profile: Conservative risk tolerance with low savings behaviour. "
            "Ultra-cautious investor who needs the cheapest and most stable funds. "
            "Money market funds, ultra-short bond ETFs, and HYSA are priorities. "
            "Suitable asset allocation: 10-30% stocks, 70-90% bonds."
        ),
    }
    for (risk, spend), text in profile_texts.items():
        docs.append({
            'text': text,
            'metadata': {'doc_type': 'profile', 'risk': risk, 'spending': spend},
        })

    # ── 2. Fund descriptions (from Kaggle dataset) ──
    name_col = next(
        (c for c in ['fund_name', 'fund_long_name', 'fund_symbol', 'symbol', 'name']
         if c in fund_feat.columns), None
    )
    fund_count = 0
    for idx, row in fund_feat.iterrows():
        fn = row.get(name_col, f"Fund_{idx}") if name_col else f"Fund_{idx}"
        parts = [f"Fund: {fn}."]
        if pd.notna(row.get('investment_type')):
            parts.append(f"Type: {row['investment_type']}.")
        if pd.notna(row.get('fund_risk_tier')):
            parts.append(f"Risk tier: {row['fund_risk_tier']}.")
        if pd.notna(row.get('expense_ratio')):
            parts.append(f"Expense ratio: {row['expense_ratio']:.4f}.")
        if pd.notna(row.get('avg_return')):
            parts.append(f"Average return: {row['avg_return']:.2f}%.")
        if pd.notna(row.get('asset_class_derived')):
            parts.append(f"Asset class: {row['asset_class_derived']}.")
        if pd.notna(row.get('alloc_stocks')):
            parts.append(f"Stock allocation: {row['alloc_stocks']*100:.0f}%.")
        if pd.notna(row.get('alloc_bonds')):
            parts.append(f"Bond allocation: {row['alloc_bonds']*100:.0f}%.")
        if pd.notna(row.get('composite_score')):
            parts.append(f"Quality score: {row['composite_score']:.2f}.")

        text = " ".join(parts)
        if len(text) > 40:
            docs.append({
                'text': text,
                'metadata': {
                    'doc_type': 'fund',
                    'fund_name': str(fn),
                    'risk_tier': str(row.get('fund_risk_tier', 'unknown')),
                },
            })
            fund_count += 1

    # ── 3. Recommendation rationales ──
    rationale_texts = [
        (
            "Why aggressive investors get high-risk funds: Aggressive profiles have "
            "higher risk tolerance scores and longer investment horizons. The cosine "
            "similarity algorithm matches them with funds that have high beta, high "
            "returns, and higher equity allocation. High savers in this group can "
            "afford funds with moderate expense ratios, while low savers are matched "
            "with cheaper index ETFs that still offer growth."
        ),
        (
            "Why moderate investors get balanced funds: Moderate profiles seek a "
            "balance of growth and stability. They are matched with balanced funds "
            "that have medium risk scores, moderate returns, and a mix of stocks "
            "and bonds. Diversification is weighted higher for this group."
        ),
        (
            "Why conservative investors get low-risk funds: Conservative profiles "
            "prioritise capital preservation. They are matched with funds that have "
            "low beta, high return consistency, high diversification, and lower "
            "volatility. Bond-heavy and treasury funds rank highest for this group."
        ),
    ]
    for text in rationale_texts:
        docs.append({'text': text, 'metadata': {'doc_type': 'rationale'}})

    # ── 4. Financial education documents ──
    education_docs = [
        (
            "Roth IRA: Contributions are made with after-tax dollars, but all growth "
            "and qualified withdrawals in retirement are completely tax-free. "
            "2024 contribution limit: $7,000 per year ($8,000 if age 50 or older). "
            "Income eligibility: Single filers phase out at $146,000-$161,000 MAGI. "
            "Married filing jointly phase out at $230,000-$240,000. "
            "If over the limit, consider a Backdoor Roth conversion strategy. "
            "Best for: younger investors in lower tax brackets who expect higher "
            "income in retirement."
        ),
        (
            "Traditional IRA: Contributions may be tax-deductible, reducing your "
            "taxable income in the year you contribute. Growth is tax-deferred. "
            "Withdrawals in retirement are taxed as ordinary income. "
            "2024 contribution limit: $7,000 ($8,000 if age 50+). "
            "If you are covered by an employer retirement plan, the deduction "
            "phases out at higher incomes ($87,000 single, $136,000 married). "
            "Best for: investors who want an immediate tax break and expect to be "
            "in a lower tax bracket in retirement."
        ),
        (
            "401(k) retirement plan: Employer-sponsored defined contribution plan. "
            "2024 employee contribution limit: $23,000 ($30,500 if age 50+). "
            "Many employers offer matching contributions — for example, matching "
            "50% of your contributions up to 6% of salary. Always contribute enough "
            "to get the full employer match — it is essentially free money. "
            "Traditional 401(k) uses pre-tax dollars; Roth 401(k) uses after-tax. "
            "Funds available are limited to what your employer plan offers, "
            "typically including target-date funds, index funds, and bond funds."
        ),
        (
            "HSA (Health Savings Account): The only account with a triple tax "
            "advantage — contributions are tax-deductible, growth is tax-free, "
            "and withdrawals for qualified medical expenses are tax-free. "
            "Requires enrollment in a High-Deductible Health Plan (HDHP). "
            "2024 contribution limits: $4,150 for individual, $8,300 for family. "
            "After age 65, funds can be withdrawn for any purpose (taxed like "
            "Traditional IRA if not for medical expenses). Many financial planners "
            "consider HSA the best retirement account available."
        ),
        (
            "Emergency fund: Financial experts recommend saving 3 to 6 months of "
            "living expenses in a liquid, easily accessible account before investing. "
            "A high-yield savings account (HYSA) is the best vehicle — currently "
            "offering 4-5% APY with no risk. This protects against job loss, "
            "medical emergencies, or unexpected expenses without needing to sell "
            "investments at a loss. Building an emergency fund should be the first "
            "financial priority before any investing."
        ),
        (
            "Asset allocation by age: A common guideline is to subtract your age "
            "from 110 to determine your stock allocation percentage. A 30-year-old "
            "would hold 80% stocks and 20% bonds. A 50-year-old would hold 60% "
            "stocks and 40% bonds. As you approach retirement, gradually shift "
            "toward bonds and stable value funds to reduce portfolio volatility. "
            "Target-date funds automate this rebalancing for you."
        ),
        (
            "Expense ratio explained: The annual fee a fund charges as a percentage "
            "of your invested assets. For example, a 0.50% expense ratio means you "
            "pay $50 per year for every $10,000 invested. Index funds and ETFs "
            "typically charge below 0.20%. Actively managed mutual funds may charge "
            "0.50% to 1.50% or more. Over 30 years of investing, a 1% difference "
            "in expense ratio can cost you over 25% of your total returns due to "
            "the compounding effect. Always compare expense ratios when choosing funds."
        ),
        (
            "Tax-efficient fund placement: Different account types have different tax "
            "treatments, so placing the right funds in the right accounts matters. "
            "Tax-inefficient investments (high-dividend funds, actively managed funds, "
            "REITs, bond funds) should go in tax-advantaged accounts like 401(k) or "
            "IRA where gains are sheltered. Tax-efficient investments (index ETFs, "
            "growth stocks with low dividends) should go in taxable brokerage accounts "
            "since they generate minimal taxable events."
        ),
        (
            "Dollar-cost averaging (DCA): Instead of investing a lump sum all at once, "
            "DCA means investing a fixed amount at regular intervals (e.g., $500/month). "
            "This reduces the impact of market volatility because you buy more shares "
            "when prices are low and fewer when prices are high. DCA is especially "
            "useful for beginners who are nervous about market timing. Most 401(k) "
            "contributions are already a form of DCA since they come from each paycheck."
        ),
    ]
    for text in education_docs:
        docs.append({'text': text, 'metadata': {'doc_type': 'education'}})

    print(f"[RAG] Built {len(docs)} documents "
          f"({len(profile_texts)} profiles, {fund_count} funds, "
          f"{len(rationale_texts)} rationales, {len(education_docs)} education)")
    return docs


# ─────────────────────────────────────────────
# 2. INITIALISE VECTOR STORE + QA CHAIN
# ─────────────────────────────────────────────

def init_rag(documents: list, persist_dir: str = "data/chromadb"):
    """
    Embed documents → store in ChromaDB → build RetrievalQA chain.

    Uses:
      - sentence-transformers (all-MiniLM-L6-v2) for embeddings — FREE, runs locally
      - ChromaDB for vector storage — FREE, runs locally
      - Google Gemini 2.5 Flash for generation — FREE API key from ai.google.dev

    Caches everything so it only runs once per app session.
    """
    if _rag_cache.get('qa_chain'):
        return _rag_cache

    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain.chains import RetrievalQA
        from langchain.schema import Document
        from langchain.prompts import PromptTemplate
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as e:
        print(f"[RAG] Missing dependency: {e}")
        print("[RAG] Run: pip install langchain langchain-google-genai "
              "langchain-community chromadb sentence-transformers")
        return None

    # ── Convert to LangChain Documents ──
    lc_docs = [
        Document(page_content=d['text'], metadata=d['metadata'])
        for d in documents
    ]

    # ── Chunk (most docs are already short but some fund descriptions may be long) ──
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(lc_docs)
    print(f"[RAG] {len(chunks)} chunks created from {len(lc_docs)} documents")

    # ── Embed with sentence-transformers (free, local) ──
    print("[RAG] Loading embedding model (all-MiniLM-L6-v2) …")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
    )
    print("[RAG] Embedding model ready")

    # ── Store in ChromaDB ──
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="financial_advisor",
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},  # retrieve top 5 relevant chunks
    )
    print(f"[RAG] ChromaDB ready — {len(chunks)} chunks indexed")

    # ── Build QA chain with Google Gemini (free) ──
    api_key = os.environ.get("GOOGLE_API_KEY")
    qa_chain = None

    if api_key:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            google_api_key=api_key,
        )

        # Custom prompt template for better financial advice answers
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a helpful financial advisor chatbot. Use the following "
                "context to answer the user's question. If you don't know the "
                "answer from the context, say so honestly. Be concise and "
                "practical. Give specific advice when possible.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            ),
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template},
        )
        print("[RAG] QA chain ready (Google Gemini 2.5 Flash)")
    else:
        print("[RAG] ⚠ GOOGLE_API_KEY not set — retriever works, QA chain disabled")
        print("[RAG]   Get a free key at https://ai.google.dev")

    _rag_cache.update({
        'vectorstore': vectorstore,
        'retriever': retriever,
        'qa_chain': qa_chain,
        'embeddings': embeddings,
    })
    return _rag_cache


# ─────────────────────────────────────────────
# 3. QUERY THE RAG PIPELINE
# ─────────────────────────────────────────────

def ask(question: str, user_context: str = "") -> dict:
    """
    Ask a question to the financial advisor chatbot.

    Args:
        question:     the user's question
        user_context: optional string describing the user's profile
                      (injected as part of the query for personalisation)

    Returns:
        dict with 'answer' (str) and 'sources' (list[dict])
    """
    if not _rag_cache.get('qa_chain'):
        return {
            'answer': (
                "The AI chatbot is not available. Please make sure:\n"
                "1. RAG dependencies are installed (`pip install langchain "
                "langchain-google-genai langchain-community chromadb "
                "sentence-transformers`)\n"
                "2. Your free Google Gemini API key is set "
                "(get one at https://ai.google.dev)"
            ),
            'sources': [],
        }

    # Prepend user context to the question for personalised answers
    full_question = question
    if user_context:
        full_question = (
            f"User context: {user_context}\n\n"
            f"Question: {question}\n\n"
            "Provide a personalised answer based on the user's profile "
            "and the retrieved context."
        )

    try:
        result = _rag_cache['qa_chain'].invoke({"query": full_question})

        sources = []
        for doc in result.get('source_documents', []):
            sources.append({
                'type': doc.metadata.get('doc_type', ''),
                'snippet': doc.page_content[:120] + '…',
            })

        return {
            'answer': result['result'],
            'sources': sources,
        }
    except Exception as e:
        return {
            'answer': f"Sorry, I encountered an error: {str(e)}",
            'sources': [],
        }
