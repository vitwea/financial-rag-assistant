"""
app.py
------
Streamlit UI for the Financial RAG Assistant.

Run:
    streamlit run app.py
"""

import time
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Financial RAG Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

  :root {
    --bg:        #0d1117;
    --surface:   #161b22;
    --border:    #30363d;
    --accent:    #e6b450;
    --accent2:   #58a6ff;
    --text:      #e6edf3;
    --muted:     #8b949e;
    --tesla:     #e31937;
    --microsoft: #00a4ef;
    --success:   #3fb950;
    --warning:   #d29922;
  }

  .stApp { background: var(--bg); color: var(--text); font-family: 'DM Sans', sans-serif; }
  .main .block-container { padding: 2rem 2rem 4rem; max-width: 1100px; }

  /* Hide sidebar and chrome */
  [data-testid="stSidebar"],
  [data-testid="collapsedControl"],
  [data-testid="InputInstructions"],
  .stDeployButton { display: none !important; }
  #MainMenu, footer, header { visibility: hidden; }

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

  /* Messages */
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

  /* Sources */
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

  /* Input */
  .stTextInput input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
  }
  .stTextInput input:focus {
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
    transition: opacity 0.15s !important;
  }
  .stButton button:hover { opacity: 0.85 !important; }

  /* Selectbox */
  .stSelectbox div[data-baseweb="select"] > div {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
  }

  hr { border-color: var(--border); margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Pipeline ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_pipeline():
    from src.pipeline.pipeline import RAGPipeline
    return RAGPipeline()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _score_color(score: float) -> str:
    if score >= 0.80:
        return "#3fb950"
    if score >= 0.60:
        return "#d29922"
    return "#e31937"


def render_sources(sources: str) -> str:
    if not sources:
        return ""
    lines = sources.strip().split("\n")
    body  = "\n".join(
        f"<div>{line}</div>" for line in lines
        if line.strip() and line.strip() != "**Sources used:**"
    )
    return f'<div class="sources-box"><strong>Sources used</strong>{body}</div>'


def render_evaluation_html(evaluation) -> str | None:
    """Return a self-contained HTML block for the eval scores, or None."""
    if evaluation is None:
        return None
    try:
        dims = [
            ("grounding",    evaluation.grounding),
            ("relevance",    evaluation.relevance),
            ("faithfulness", evaluation.faithfulness),
            ("completeness", evaluation.completeness),
        ]
    except AttributeError:
        return None

    rows = ""
    for label, score in dims:
        pct   = int(score * 100)
        color = _score_color(score)
        rows += f"""
        <div style="display:flex;align-items:center;gap:10px;margin:4px 0;
                    font-size:13px;font-family:monospace;">
          <span style="color:#8b949e;width:100px;flex-shrink:0;">{label}</span>
          <div style="flex:1;height:5px;background:#30363d;border-radius:10px;overflow:hidden;">
            <div style="width:{pct}%;height:100%;background:{color};border-radius:10px;"></div>
          </div>
          <span style="color:#e6edf3;width:36px;text-align:right;flex-shrink:0;">{score:.2f}</span>
        </div>"""

    status_color = "#3fb950" if evaluation.passed else "#e31937"
    status_txt   = f"✅ PASS — avg {evaluation.average:.2f}" if evaluation.passed \
                   else f"❌ FAIL — avg {evaluation.average:.2f}"

    return f"""<!DOCTYPE html>
<html>
<body style="margin:0;padding:0;background:transparent;">
  <div style="background:rgba(0,0,0,0.25);border:1px solid #30363d;border-radius:8px;
              padding:12px 14px;font-family:'DM Sans',sans-serif;">
    <div style="font-size:11px;color:#8b949e;margin-bottom:10px;
                letter-spacing:0.06em;text-transform:uppercase;">
      Claude Judge Evaluation
    </div>
    {rows}
    <div style="font-size:12px;font-weight:600;margin-top:8px;color:{status_color};">
      {status_txt}
    </div>
  </div>
</body>
</html>"""


# ── Session state ─────────────────────────────────────────────────────────────

if "messages"  not in st.session_state:
    st.session_state["messages"]  = []
if "input_key" not in st.session_state:
    st.session_state["input_key"] = 0


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="rag-header">
  <h1>📊 Financial RAG Assistant</h1>
  <p>Ask questions about Tesla, Apple & Microsoft annual reports —
     every answer is grounded in real 10-K passages with page-level citations.</p>
</div>
""", unsafe_allow_html=True)


# ── Filters row ───────────────────────────────────────────────────────────────

fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 4])

with fc1:
    company_filter = st.selectbox(
        "Company", ["All companies", "Tesla", "Apple", "Microsoft"],
    )
    company_filter_val = None if company_filter == "All companies" \
                         else company_filter.lower()

with fc2:
    year_filter = st.selectbox(
        "Year", ["All years", "2025", "2024", "2023", "2022"],
    )
    year_filter_val = None if year_filter == "All years" else int(year_filter)

with fc3:
    evaluate = st.toggle(
        "AI evaluation", value=False,
        help="Uses Claude as a judge to score each response (adds ~3s latency)",
    )

with fc4:
    if st.button("🗑️ Clear conversation"):
        st.session_state["messages"] = []
        st.rerun()

st.markdown("<hr>", unsafe_allow_html=True)


# ── Conversation ──────────────────────────────────────────────────────────────

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
            latency = f'<span class="latency-pill">{msg.get("latency_ms")}ms</span>' \
                      if msg.get("latency_ms") else ""
            st.markdown(
                f'<div class="msg-assistant">🤖 {latency}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(msg["content"])

            sources_html = render_sources(msg.get("sources", ""))
            if sources_html:
                st.markdown(sources_html, unsafe_allow_html=True)

            eval_html = render_evaluation_html(msg.get("evaluation"))
            if eval_html:
                components.html(eval_html, height=140, scrolling=False)


# ── Input ─────────────────────────────────────────────────────────────────────

st.markdown("<div style='margin-top:2rem'></div>", unsafe_allow_html=True)

col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_input(
        "query",
        label_visibility="collapsed",
        placeholder="Ask about Tesla, Apple, or Microsoft annual reports...",
        key=f"qi_{st.session_state['input_key']}",
    )
with col2:
    submit = st.button("Ask →", use_container_width=True)


# ── Submit ────────────────────────────────────────────────────────────────────

if submit and query.strip():
    st.session_state["input_key"] += 1
    st.session_state["messages"].append({"role": "user", "content": query})

    with st.spinner("Searching 10-K documents..."):
        try:
            pipeline   = get_pipeline()
            t0         = time.perf_counter()
            result     = pipeline.ask(
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