"""
app.py
------
Streamlit UI for the Financial RAG Assistant.

Provides a clean, professional chat interface for querying
Tesla, Apple, and Microsoft 10-K annual reports.

Run:
    streamlit run src/app.py
"""

import time
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────

st.set_page_config(
    page_title="Financial RAG Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* Import fonts */
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

  /* Root variables */
  :root {
    --bg:        #0d1117;
    --surface:   #161b22;
    --border:    #30363d;
    --accent:    #e6b450;
    --accent2:   #58a6ff;
    --text:      #e6edf3;
    --muted:     #8b949e;
    --tesla:     #e31937;
    --apple:     #555555;
    --microsoft: #00a4ef;
    --success:   #3fb950;
    --warning:   #d29922;
  }

  /* Global */
  .stApp { background: var(--bg); color: var(--text); font-family: 'DM Sans', sans-serif; }
  .main .block-container { padding: 2rem 2rem 4rem; max-width: 1100px; }

  /* Header */
  .rag-header {
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.5rem;
    margin-bottom: 2rem;
  }
  .rag-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--accent);
    margin: 0;
    letter-spacing: -0.02em;
  }
  .rag-header p {
    color: var(--muted);
    font-size: 0.95rem;
    margin: 0.4rem 0 0;
    font-weight: 300;
  }

  /* Company badges */
  .badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
    margin-right: 6px;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.03em;
  }
  .badge-tesla     { background: rgba(227,25,55,0.15); color: var(--tesla); border: 1px solid rgba(227,25,55,0.3); }
  .badge-apple     { background: rgba(85,85,85,0.15);  color: #aaa;        border: 1px solid rgba(85,85,85,0.3); }
  .badge-microsoft { background: rgba(0,164,239,0.15); color: var(--microsoft); border: 1px solid rgba(0,164,239,0.3); }

  /* Chat messages */
  .msg-user {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px 12px 4px 12px;
    padding: 1rem 1.2rem;
    margin: 1rem 0 0.5rem 4rem;
    font-size: 0.95rem;
    line-height: 1.6;
  }
  .msg-assistant {
    background: rgba(230,180,80,0.04);
    border: 1px solid rgba(230,180,80,0.15);
    border-radius: 12px 12px 12px 4px;
    padding: 1.2rem 1.4rem;
    margin: 0.5rem 4rem 0 0;
    font-size: 0.92rem;
    line-height: 1.7;
  }
  .msg-blocked {
    background: rgba(210,153,34,0.07);
    border: 1px solid rgba(210,153,34,0.2);
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    margin: 0.5rem 4rem 0 0;
    font-size: 0.9rem;
    color: var(--warning);
  }

  /* Sources section */
  .sources-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-top: 0.8rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
    line-height: 1.8;
  }
  .sources-box strong { color: var(--text); font-weight: 500; }

  /* Eval scores */
  .eval-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 3px 0;
    font-size: 0.8rem;
    font-family: 'DM Mono', monospace;
  }
  .eval-label { color: var(--muted); width: 100px; flex-shrink: 0; }
  .eval-track {
    flex: 1;
    height: 5px;
    background: var(--border);
    border-radius: 10px;
    overflow: hidden;
  }
  .eval-fill  { height: 100%; border-radius: 10px; }
  .eval-value { color: var(--text); width: 36px; text-align: right; flex-shrink: 0; }
  .eval-pass  { color: var(--success); font-size: 0.78rem; font-weight: 500; margin-top: 4px; }
  .eval-fail  { color: var(--tesla);   font-size: 0.78rem; font-weight: 500; margin-top: 4px; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
  }
  [data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    color: var(--accent);
  }

  /* Input */
  .stTextInput input, .stTextArea textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
  }
  .stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(230,180,80,0.15) !important;
  }

  /* Buttons */
  .stButton button {
    background: var(--accent) !important;
    color: #0d1117 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 0.01em !important;
    transition: opacity 0.15s !important;
  }
  .stButton button:hover { opacity: 0.85 !important; }

  /* Selectbox */
  .stSelectbox div[data-baseweb="select"] > div {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
  }

  /* Latency pill */
  .latency-pill {
    display: inline-block;
    background: rgba(88,166,255,0.1);
    border: 1px solid rgba(88,166,255,0.2);
    color: var(--accent2);
    border-radius: 20px;
    padding: 1px 10px;
    font-size: 0.75rem;
    font-family: 'DM Mono', monospace;
    margin-left: 8px;
    vertical-align: middle;
  }

  /* Divider */
  hr { border-color: var(--border); margin: 1.5rem 0; }

  /* Hide Streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Pipeline initialisation (cached) ─────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_pipeline():
    """Load the RAG pipeline once and reuse across all sessions."""
    from src.pipeline.pipeline import RAGPipeline
    return RAGPipeline()


# ── Helper: render evaluation scores ─────────────────────────────────────────

def _score_color(score: float) -> str:
    if score >= 0.80:
        return "#3fb950"
    if score >= 0.60:
        return "#d29922"
    return "#e31937"


def render_evaluation(evaluation) -> str:
    if evaluation is None:
        return ""

    # Guard: make sure it's a proper evaluation object
    try:
        dims = [
            ("grounding",    evaluation.grounding),
            ("relevance",    evaluation.relevance),
            ("faithfulness", evaluation.faithfulness),
            ("completeness", evaluation.completeness),
        ]
    except AttributeError:
        return ""

    rows = ""
    for label, score in dims:
        pct   = int(score * 100)
        color = _score_color(score)
        rows += f"""
        <div style="display:flex;align-items:center;gap:10px;margin:4px 0;font-size:0.8rem;font-family:'DM Mono',monospace;">
          <span style="color:#8b949e;width:100px;flex-shrink:0;">{label}</span>
          <div style="flex:1;height:5px;background:#30363d;border-radius:10px;overflow:hidden;">
            <div style="width:{pct}%;height:100%;background:{color};border-radius:10px;"></div>
          </div>
          <span style="color:#e6edf3;width:36px;text-align:right;flex-shrink:0;">{score:.2f}</span>
        </div>"""

    status_color = "#3fb950" if evaluation.passed else "#e31937"
    status_txt   = f"✅ PASS — avg {evaluation.average:.2f}" if evaluation.passed \
                   else f"❌ FAIL — avg {evaluation.average:.2f}"

    return f"""
    <div style="margin-top:0.8rem;padding:0.8rem 1rem;background:rgba(0,0,0,0.2);
                border-radius:8px;border:1px solid #30363d;">
      <div style="font-size:0.72rem;color:#8b949e;margin-bottom:8px;
                  font-family:'DM Mono',monospace;letter-spacing:0.05em;">
        CLAUDE JUDGE EVALUATION
      </div>
      {rows}
      <div style="font-size:0.78rem;font-weight:500;margin-top:6px;color:{status_color};">
        {status_txt}
      </div>
    </div>"""


# ── Helper: render sources ────────────────────────────────────────────────────

def render_sources(sources: str) -> str:
    if not sources:
        return ""
    lines = sources.strip().split("\n")
    body  = "\n".join(
        f"<div>{line}</div>" for line in lines
        if line.strip() and line.strip() != "**Sources used:**"
    )
    return f"""
    <div class="sources-box">
      <strong>Sources used</strong>
      {body}
    </div>"""


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")

    company_filter = st.selectbox(
        "Company filter",
        options=["All companies", "Tesla", "Apple", "Microsoft"],
        help="Restrict retrieval to a single company's 10-K",
    )
    company_filter_val = None if company_filter == "All companies" \
                         else company_filter.lower()

    year_filter = st.selectbox(
        "Year filter",
        options=["All years", "2025", "2024", "2023", "2022"],
        help="Restrict retrieval to a specific fiscal year",
    )
    year_filter_val = None if year_filter == "All years" else int(year_filter)

    evaluate = st.toggle(
        "Enable AI evaluation",
        value=False,
        help="Uses Claude as a judge to score each response (adds ~3s latency)",
    )

    st.markdown("---")
    st.markdown("### 📚 Data sources")
    st.markdown("""
<span class="badge badge-tesla">Tesla</span> FY2024 10-K<br><br>
<span class="badge badge-apple">Apple</span> FY2025 10-K<br><br>
<span class="badge badge-microsoft">Microsoft</span> FY2025 10-K
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💡 Example queries")
    examples = [
        "What are Tesla's main risk factors?",
        "Compare cloud strategies of Apple and Microsoft",
        "How has Microsoft Azure revenue evolved over the years?",
        "How has Tesla described AI risks historically?",
        "Compare R&D spending across all three companies",
        "How did Apple describe iPhone revenue in 2022 vs 2025?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:20]}", use_container_width=True):
            st.session_state["pending_query"] = ex

    st.markdown("---")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()


# ── Main layout ───────────────────────────────────────────────────────────────

st.markdown("""
<div class="rag-header">
  <h1>📊 Financial RAG Assistant</h1>
  <p>Ask questions about Tesla, Apple & Microsoft annual reports — every answer is grounded in real 10-K passages with page-level citations.</p>
</div>
""", unsafe_allow_html=True)

# Session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ── Render conversation history ───────────────────────────────────────────────

for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="msg-user">🧑 {msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        if msg.get("blocked"):
            st.markdown(
                f'<div class="msg-blocked">🛡️ {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            latency = f'<span class="latency-pill">{msg.get("latency_ms", 0)}ms</span>' \
                      if msg.get("latency_ms") else ""
            # Header with latency pill
            st.markdown(
                f'<div class="msg-assistant">🤖 {latency}</div>',
                unsafe_allow_html=True,
            )
            # Answer as native markdown (avoids HTML escaping issues)
            st.markdown(msg["content"])
            # Sources box
            sources_html = render_sources(msg.get("sources", ""))
            if sources_html:
                st.markdown(sources_html, unsafe_allow_html=True)
            # Evaluation bars
            eval_html = render_evaluation(msg.get("evaluation"))
            if eval_html:
                st.markdown(eval_html, unsafe_allow_html=True)


# ── Query input ───────────────────────────────────────────────────────────────

st.markdown("<div style='margin-top:2rem'></div>", unsafe_allow_html=True)

# Pre-fill from sidebar example buttons
default_query = st.session_state.pop("pending_query", "")

col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_input(
        label="query",
        label_visibility="collapsed",
        placeholder="Ask about Tesla, Apple, or Microsoft annual reports...",
        value=default_query,
        key="query_input",
    )
with col2:
    submit = st.button("Ask →", use_container_width=True)


# ── Handle submission ─────────────────────────────────────────────────────────

if submit and query.strip():
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": query})

    # Run pipeline
    with st.spinner("Searching 10-K documents..."):
        try:
            pipeline  = get_pipeline()
            t0        = time.perf_counter()
            result    = pipeline.ask(
                query,
                company_filter=company_filter_val,
                year_filter=year_filter_val,
                evaluate=evaluate,
            )
            latency_ms = int((time.perf_counter() - t0) * 1000)

            st.session_state["messages"].append({
                "role":       "assistant",
                "content":    result["answer"],
                "sources":    result.get("sources", ""),
                "evaluation": result.get("evaluation"),
                "latency_ms": latency_ms,
                "blocked":    result.get("blocked", False),
            })

        except Exception as exc:
            st.session_state["messages"].append({
                "role":    "assistant",
                "content": f"An error occurred: {exc}",
                "blocked": True,
            })

    st.rerun()