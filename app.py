"""
app.py  —  Financial RAG Assistant
"""

import time
import streamlit as st

st.set_page_config(
    page_title="Financial RAG Assistant",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg:        #1c1c1e;
  --surface:   #2c2c2e;
  --surface2:  #3a3a3c;
  --border:    #3a3a3c;
  --text:      #f2f2f7;
  --muted:     #8e8e93;
  --user-bg:   #0a84ff;
  --bot-bg:    #2c2c2e;
  --tesla:     #ff453a;
  --apple:     #aeaeb2;
  --msft:      #0a84ff;
  --pass:      #30d158;
  --fail:      #ff453a;
  --warn:      #ffd60a;
}

* { box-sizing: border-box; }
.stApp { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; }
.main .block-container { padding: 0 1rem 6rem; max-width: 720px; }

/* ── HEADER ── */
.app-header {
  text-align: center;
  padding: 2.5rem 1rem 1.5rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1.5rem;
}
.app-title {
  font-size: 1.3rem;
  font-weight: 600;
  color: var(--text);
  letter-spacing: -0.01em;
  margin: 0 0 5px;
}
.app-sub {
  font-size: 0.82rem;
  color: var(--muted);
  font-weight: 300;
  margin: 0 0 12px;
}
.app-tags {
  display: flex;
  gap: 5px;
  justify-content: center;
  flex-wrap: wrap;
}
.tag {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem;
  padding: 2px 8px;
  border-radius: 4px;
  border: 1px solid var(--border);
  color: var(--muted);
  background: var(--surface);
}
.tag-tesla { color: var(--tesla); border-color: rgba(255,69,58,0.3); background: rgba(255,69,58,0.07); }
.tag-apple { color: var(--apple); border-color: rgba(174,174,178,0.2); background: rgba(174,174,178,0.06); }
.tag-msft  { color: var(--msft);  border-color: rgba(10,132,255,0.3); background: rgba(10,132,255,0.07); }

/* ── CHAT MESSAGES ── */
.chat-row {
  display: flex;
  margin: 0.6rem 0;
  align-items: flex-end;
  gap: 8px;
  animation: fadeIn 0.2s ease;
}
.chat-row.user  { flex-direction: row-reverse; }
.chat-row.bot   { flex-direction: row; }

.bubble {
  max-width: 78%;
  padding: 0.7rem 1rem;
  border-radius: 18px;
  font-size: 0.9rem;
  line-height: 1.6;
}
.bubble.user {
  background: var(--user-bg);
  color: #ffffff;
  border-radius: 18px 18px 4px 18px;
}
.bubble.bot {
  background: var(--bot-bg);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 18px 18px 18px 4px;
}
.bubble.blocked {
  background: rgba(255,214,10,0.06);
  border: 1px solid rgba(255,214,10,0.2);
  border-radius: 18px 18px 18px 4px;
  color: var(--warn);
  font-size: 0.87rem;
  max-width: 78%;
}

.meta {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.62rem;
  color: var(--muted);
  margin-bottom: 3px;
  padding: 0 4px;
  display: flex;
  gap: 6px;
}
.meta.user { justify-content: flex-end; }

/* ── SOURCES ── */
.sources-wrap {
  max-width: 78%;
  margin: 2px 0 0 0;
  padding: 0.55rem 0.9rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  font-family: 'JetBrains Mono', monospace;
}
.sources-label {
  font-size: 0.6rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--muted);
  margin-bottom: 6px;
}
.src-chips { display: flex; flex-wrap: wrap; gap: 4px; }
.src-chip {
  font-size: 0.65rem;
  padding: 2px 7px;
  border-radius: 4px;
  border: 1px solid;
  white-space: nowrap;
}
.src-tesla { color: var(--tesla); border-color: rgba(255,69,58,0.3); background: rgba(255,69,58,0.07); }
.src-apple { color: var(--apple); border-color: rgba(174,174,178,0.2); background: rgba(174,174,178,0.06); }
.src-msft  { color: var(--msft);  border-color: rgba(10,132,255,0.3); background: rgba(10,132,255,0.07); }
.src-def   { color: var(--muted); border-color: var(--border); background: var(--surface2); }

/* ── EVAL ── */
.eval-wrap {
  max-width: 78%;
  margin: 2px 0 0 0;
  padding: 0.6rem 0.9rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
}
.eval-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.6rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--muted);
  margin-bottom: 8px;
}
.eval-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 4px 0;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.68rem;
}
.eval-dim   { color: var(--muted); width: 86px; flex-shrink: 0; }
.eval-track { flex: 1; height: 3px; background: var(--surface2); border-radius: 2px; overflow: hidden; }
.eval-fill  { height: 100%; border-radius: 2px; }
.eval-num   { color: var(--text); width: 28px; text-align: right; flex-shrink: 0; }
.eval-verdict {
  margin-top: 6px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.68rem;
  font-weight: 500;
}
.v-pass { color: var(--pass); }
.v-fail { color: var(--fail); }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
  background: #111113 !important;
  border-right: 1px solid var(--border) !important;
}
.sidebar-title {
  font-size: 0.9rem;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 1rem;
}
.sidebar-sect {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.6rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--muted);
  margin: 1rem 0 0.5rem;
}
.cov-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 3px 0;
  font-size: 0.8rem;
  color: var(--text);
}
.cov-years {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem;
  color: var(--muted);
}
.dot { width: 6px; height: 6px; border-radius: 50%; margin-right: 7px; display: inline-block; }

/* ── INPUT ── */
.stTextInput input {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 22px !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.92rem !important;
  padding: 0.65rem 1.1rem !important;
}
.stTextInput input:focus {
  border-color: var(--msft) !important;
  box-shadow: 0 0 0 3px rgba(10,132,255,0.15) !important;
}
.stButton button {
  background: var(--msft) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 22px !important;
  font-weight: 600 !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.9rem !important;
}
.stButton button:hover { opacity: 0.88 !important; }
.stSelectbox div[data-baseweb="select"] > div {
  background: var(--surface) !important;
  border-color: var(--border) !important;
  color: var(--text) !important;
  border-radius: 8px !important;
  font-size: 0.87rem !important;
}

/* ── MISC ── */
@keyframes fadeIn { from { opacity:0; transform:translateY(4px); } to { opacity:1; transform:translateY(0); } }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
hr { border-color: var(--border); }
</style>
""", unsafe_allow_html=True)


# ── Pipeline ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_pipeline():
    from src.pipeline.pipeline import RAGPipeline
    return RAGPipeline()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _score_color(s: float) -> str:
    if s >= 0.80: return "#30d158"
    if s >= 0.60: return "#ffd60a"
    return "#ff453a"

def _src_class(s: str) -> str:
    s = s.lower()
    if "tesla"     in s: return "src-tesla"
    if "apple"     in s: return "src-apple"
    if "microsoft" in s: return "src-msft"
    return "src-def"

def render_sources(sources: str) -> str:
    if not sources:
        return ""
    lines = [l.strip().lstrip("•·- ") for l in sources.strip().split("\n")
             if l.strip() and "Sources used" not in l]
    chips = "".join(
        f'<span class="src-chip {_src_class(l)}">{l}</span>'
        for l in lines if l
    )
    if not chips:
        return ""
    return f"""
    <div class="sources-wrap">
      <div class="sources-label">Sources</div>
      <div class="src-chips">{chips}</div>
    </div>"""

def render_evaluation(evaluation) -> str:
    if evaluation is None:
        return ""
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
        color = _score_color(score)
        rows += f"""
        <div class="eval-row">
          <span class="eval-dim">{label}</span>
          <div class="eval-track"><div class="eval-fill" style="width:{int(score*100)}%;background:{color};"></div></div>
          <span class="eval-num">{score:.2f}</span>
        </div>"""
    v_cls = "v-pass" if evaluation.passed else "v-fail"
    v_txt = f"✓ pass · avg {evaluation.average:.2f}" if evaluation.passed \
            else f"✗ fail · avg {evaluation.average:.2f}"
    return f"""
    <div class="eval-wrap">
      <div class="eval-label">Evaluation</div>
      {rows}
      <div class="eval-verdict {v_cls}">{v_txt}</div>
    </div>"""


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sidebar-title">⚙️ Settings</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="sidebar-sect">Company</div>', unsafe_allow_html=True)
    company_filter = st.selectbox(
        "company", label_visibility="collapsed",
        options=["All companies", "Tesla", "Apple", "Microsoft"],
    )
    company_filter_val = None if company_filter == "All companies" else company_filter.lower()

    st.markdown('<div class="sidebar-sect">Year</div>', unsafe_allow_html=True)
    year_filter = st.selectbox(
        "year", label_visibility="collapsed",
        options=["All years", "2026", "2025", "2024", "2023", "2022"],
    )
    year_filter_val = None if year_filter == "All years" else int(year_filter)

    st.markdown('<div style="margin-top:0.5rem"></div>', unsafe_allow_html=True)
    evaluate = st.toggle("AI evaluation", value=False,
                         help="Claude scores each answer (+3s)")

    st.divider()

    st.markdown('<div class="sidebar-sect">Coverage</div>', unsafe_allow_html=True)
    st.markdown("""
    <div>
      <div class="cov-row"><span><span class="dot" style="background:#ff453a;"></span>Tesla</span><span class="cov-years">2022–2026</span></div>
      <div class="cov-row"><span><span class="dot" style="background:#aeaeb2;"></span>Apple</span><span class="cov-years">2022–2025</span></div>
      <div class="cov-row"><span><span class="dot" style="background:#0a84ff;"></span>Microsoft</span><span class="cov-years">2022–2025</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="sidebar-sect">Examples</div>', unsafe_allow_html=True)
    examples = [
        "What are Tesla's main risk factors?",
        "Compare cloud strategies of Apple and Microsoft",
        "How has Tesla described AI risks historically?",
        "How has Microsoft Azure revenue evolved?",
        "Compare R&D spending across all companies",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:20]}", use_container_width=True):
            st.session_state["pending_query"] = ex

    st.divider()
    if st.button("🗑 Clear chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
  <div class="app-title">📊 Financial RAG Assistant</div>
  <p class="app-sub">Ask questions about Tesla, Apple & Microsoft 10-K filings — answers grounded in real passages.</p>
  <div class="app-tags">
    <span class="tag tag-tesla">Tesla</span>
    <span class="tag tag-apple">Apple</span>
    <span class="tag tag-msft">Microsoft</span>
    <span class="tag">1,314 chunks · BGE-large · Cohere · GPT-4o-mini</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Session ───────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ── Conversation ──────────────────────────────────────────────────────────────

for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="chat-row user">
          <div class="bubble user">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        if msg.get("blocked"):
            st.markdown(f"""
            <div class="chat-row bot">
              <div class="bubble blocked">{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            latency = f'{msg["latency_ms"]}ms' if msg.get("latency_ms") else ""
            st.markdown(f"""
            <div class="meta bot">
              <span>Assistant</span>
              <span>{latency}</span>
            </div>
            <div class="chat-row bot">
              <div class="bubble bot">
            """, unsafe_allow_html=True)

            st.markdown(msg["content"])
            st.markdown("</div></div>", unsafe_allow_html=True)

            src = render_sources(msg.get("sources", ""))
            if src:
                st.markdown(src, unsafe_allow_html=True)

            evl = render_evaluation(msg.get("evaluation"))
            if evl:
                st.markdown(evl, unsafe_allow_html=True)


# ── Input ─────────────────────────────────────────────────────────────────────

st.markdown("<div style='margin-top:2rem'></div>", unsafe_allow_html=True)

default_query = st.session_state.pop("pending_query", "")
col1, col2 = st.columns([6, 1])
with col1:
    query = st.text_input(
        label="q", label_visibility="collapsed",
        placeholder="Ask about risk factors, revenue, strategy...",
        value=default_query,
        key="query_input",
    )
with col2:
    submit = st.button("Send", use_container_width=True)


# ── Submit ────────────────────────────────────────────────────────────────────

if submit and query.strip():
    st.session_state["messages"].append({"role": "user", "content": query})

    with st.spinner("Searching..."):
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
                "role": "assistant", "content": f"Error: {exc}", "blocked": True,
            })

    st.rerun()