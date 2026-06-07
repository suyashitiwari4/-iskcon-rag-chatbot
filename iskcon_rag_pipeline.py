"""
ISKCON Guru Q&A Chatbot — Streamlit Frontend
Run: streamlit run app.py
Requires: pip install streamlit chromadb sentence-transformers groq
"""
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import os
import time
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

# ── CONFIG ──────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
DB_PATH      = "iskcon_vectordb"
TOP_K        = 5

# ── PAGE CONFIG ─────────────────────────────────────────
st.set_page_config(
    page_title="ISKCON Wisdom • Chaitanya Charan Das",
    page_icon="🪷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ───────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root palette ── */
:root {
    --saffron:      #E8651A;
    --saffron-soft: #F5A35A;
    --saffron-pale: #FDF3EA;
    --gold:         #C8922A;
    --gold-pale:    #FBF0DC;
    --lotus:        #8B3A5E;
    --lotus-pale:   #F7EDF3;
    --ink:          #1C1208;
    --ink-mid:      #4A3728;
    --parchment:    #FAF6F0;
    --parchment-2:  #F2EAE0;
    --border:       rgba(200,146,42,0.25);
}

/* ── Global resets ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--parchment) !important;
    color: var(--ink) !important;
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 100% !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #2C1A0E 0%, #1C1208 60%, #0F0A05 100%) !important;
    border-right: 1px solid rgba(200,146,42,0.2);
}
[data-testid="stSidebar"] * { color: #E8D5B0 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: var(--saffron-soft) !important; }
[data-testid="stSidebar"] .stMarkdown p { color: #C4A97A !important; font-size: 0.85rem; line-height: 1.6; }
[data-testid="stSidebar"] hr { border-color: rgba(200,146,42,0.2) !important; }

/* ── Sidebar selectbox / slider ── */
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label { color: #C4A97A !important; font-size: 0.8rem; }

/* ── Main area header ── */
.iskcon-header {
    background: linear-gradient(135deg, #2C1A0E 0%, #1C1208 100%);
    padding: 2.5rem 3rem 2rem;
    margin: -1rem -1rem 2rem -1rem;
    border-bottom: 2px solid var(--gold);
    position: relative;
    overflow: hidden;
}
.iskcon-header::before {
    content: "ॐ";
    position: absolute;
    right: 3rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 7rem;
    color: rgba(200,146,42,0.08);
    font-family: serif;
    line-height: 1;
}
.iskcon-header h1 {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 2.6rem !important;
    font-weight: 300 !important;
    color: #F5E6C8 !important;
    margin: 0 0 0.3rem 0 !important;
    letter-spacing: 0.02em;
    line-height: 1.2;
}
.iskcon-header p {
    color: var(--saffron-soft) !important;
    font-size: 0.9rem;
    margin: 0;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Decorative divider ── */
.lotus-divider {
    text-align: center;
    color: var(--gold);
    font-size: 1.2rem;
    letter-spacing: 0.5rem;
    margin: 0.5rem 0 1.5rem;
    opacity: 0.6;
}

/* ── Chat messages ── */
.chat-container {
    max-width: 820px;
    margin: 0 auto;
}

.msg-user {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 1.2rem;
    animation: slideInRight 0.3s ease;
}
.msg-user .bubble {
    background: linear-gradient(135deg, var(--saffron) 0%, #C4551A 100%);
    color: #FDF6EE !important;
    padding: 0.9rem 1.3rem;
    border-radius: 18px 18px 4px 18px;
    max-width: 72%;
    font-size: 0.95rem;
    line-height: 1.6;
    box-shadow: 0 4px 16px rgba(232,101,26,0.25);
}

.msg-bot {
    display: flex;
    justify-content: flex-start;
    margin-bottom: 1.5rem;
    gap: 0.75rem;
    animation: slideInLeft 0.3s ease;
}
.msg-bot .avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: linear-gradient(135deg, #2C1A0E, #4A3728);
    border: 1.5px solid var(--gold);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
    margin-top: 2px;
}
.msg-bot .bubble {
    background: white;
    border: 1px solid var(--border);
    padding: 1.1rem 1.4rem;
    border-radius: 4px 18px 18px 18px;
    max-width: 80%;
    font-size: 0.95rem;
    line-height: 1.75;
    color: var(--ink-mid) !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.msg-bot .bubble .takeaway {
    margin-top: 0.9rem;
    padding-top: 0.8rem;
    border-top: 1px solid var(--border);
    font-style: italic;
    color: var(--lotus) !important;
    font-size: 0.88rem;
}
.msg-bot .sources {
    margin-top: 0.5rem;
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
}
.source-chip {
    background: var(--gold-pale);
    border: 1px solid rgba(200,146,42,0.3);
    color: var(--gold) !important;
    font-size: 0.72rem;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    font-family: 'DM Sans', sans-serif;
    letter-spacing: 0.02em;
}

/* ── Input area ── */
.stTextInput input {
    background: white !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 0.8rem 1.2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    color: var(--ink) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput input:focus {
    border-color: var(--saffron) !important;
    box-shadow: 0 0 0 3px rgba(232,101,26,0.12) !important;
    outline: none !important;
}
.stTextInput input::placeholder { color: #B0977A !important; }

/* ── Buttons ── */
.stButton button {
    background: linear-gradient(135deg, var(--saffron) 0%, #C4551A 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.8rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.04em !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 14px rgba(232,101,26,0.3) !important;
}
.stButton button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(232,101,26,0.4) !important;
}

/* ── Suggested questions ── */
.suggest-btn button {
    background: white !important;
    color: var(--ink-mid) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    padding: 0.5rem 0.9rem !important;
    box-shadow: none !important;
    text-align: left !important;
    transition: all 0.2s !important;
}
.suggest-btn button:hover {
    background: var(--saffron-pale) !important;
    border-color: var(--saffron-soft) !important;
    color: var(--saffron) !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Stats cards ── */
.stat-card {
    background: rgba(200,146,42,0.08);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    text-align: center;
    margin-bottom: 1rem;
}
.stat-card .num {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.8rem;
    color: var(--saffron-soft);
    line-height: 1;
    font-weight: 600;
}
.stat-card .label {
    font-size: 0.72rem;
    color: #C4A97A;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: var(--ink-mid);
}
.empty-state .om {
    font-size: 4rem;
    opacity: 0.15;
    margin-bottom: 1rem;
    font-family: serif;
}
.empty-state h3 {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.6rem;
    font-weight: 400;
    color: var(--ink-mid);
    margin-bottom: 0.5rem;
}
.empty-state p { font-size: 0.88rem; color: #8A7060; }

/* ── Spinner override ── */
.stSpinner > div { border-top-color: var(--saffron) !important; }

/* ── Animations ── */
@keyframes slideInRight {
    from { opacity: 0; transform: translateX(20px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-20px); }
    to   { opacity: 1; transform: translateX(0); }
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(200,146,42,0.3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── BACKEND ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_vector_store():
    ef  = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    col = chromadb.PersistentClient(path=DB_PATH).get_collection(
        name="iskcon_transcripts", embedding_function=ef
    )
    return col

def generate_answer(question: str, context: str) -> str:
    prompt = f"""You are a helpful assistant answering questions based on teachings of ISKCON guru Chaitanya Charan das.

Use ONLY the transcript excerpts below. Write a clear answer in 150-200 words.
Speak naturally and warmly. End with "Key takeaway: ..."

QUESTION: {question}

EXCERPTS:
{context}

ANSWER:"""
    client   = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()

def query_rag(question: str, top_k: int = TOP_K):
    col     = load_vector_store()
    results = col.query(query_texts=[question], n_results=top_k)
    chunks  = results["documents"][0]
    sources = [m["title"] for m in results["metadatas"][0]]
    context = "\n\n".join(chunks)
    answer  = generate_answer(question, context)
    unique_sources = list(dict.fromkeys(sources))[:3]
    return answer, unique_sources


# ── SESSION STATE ────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "question_count" not in st.session_state:
    st.session_state.question_count = 0
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""


# ── SIDEBAR ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🪷 ISKCON Wisdom")
    st.markdown("---")

    st.markdown("""
**About this chatbot**

Answers are drawn exclusively from the recorded discourses of **Chaitanya Charan Das** — a monk, author, and spiritual teacher in the Vaishnava tradition.
    """)

    st.markdown("---")

    # Stats
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
<div class="stat-card">
  <div class="num">1,391</div>
  <div class="label">Transcripts</div>
</div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
<div class="stat-card">
  <div class="num">{st.session_state.question_count}</div>
  <div class="label">Questions asked</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    top_k = st.slider("Chunks retrieved per query", min_value=3, max_value=10, value=5, step=1)
    st.markdown("*Higher = more context, slightly slower*")

    st.markdown("---")

    if st.button("🗑 Clear conversation"):
        st.session_state.messages = []
        st.session_state.question_count = 0
        st.rerun()

    st.markdown("---")
    st.markdown("""
<div style="font-size:0.75rem; color:#7A6550; line-height:1.6;">
Powered by <b style="color:#C4A97A">ChromaDB</b> · <b style="color:#C4A97A">all-MiniLM-L6-v2</b> · <b style="color:#C4A97A">Llama 3.3 70B</b>
</div>
""", unsafe_allow_html=True)


# ── HEADER ───────────────────────────────────────────────
st.markdown("""
<div class="iskcon-header">
  <h1>🪷 Ask the Guru</h1>
  <p>Wisdom from the discourses of Chaitanya Charan Das</p>
</div>
""", unsafe_allow_html=True)


# ── SUGGESTED QUESTIONS ──────────────────────────────────
SUGGESTIONS = [
    "What is the purpose of chanting Hare Krishna?",
    "How can one control the mind?",
    "What does the Bhagavad Gita say about duty?",
    "How to deal with grief and loss?",
    "What is the significance of devotional service?",
]

if not st.session_state.messages:
    st.markdown("""
<div class="empty-state">
  <div class="om">ॐ</div>
  <h3>Seek wisdom from the discourses</h3>
  <p>Ask any question about spiritual practice, philosophy, or daily life guidance.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("**Suggested questions:**")
    cols = st.columns(len(SUGGESTIONS))
    for i, suggestion in enumerate(SUGGESTIONS):
        with cols[i]:
            st.markdown('<div class="suggest-btn">', unsafe_allow_html=True)
            if st.button(suggestion, key=f"suggest_{i}"):
                st.session_state.pending_question = suggestion
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


# ── CHAT HISTORY ─────────────────────────────────────────
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
<div class="msg-user">
  <div class="bubble">{msg["content"]}</div>
</div>""", unsafe_allow_html=True)
    else:
        # Split answer and "Key takeaway:" line if present
        answer_text = msg["content"]
        takeaway_html = ""
        if "Key takeaway:" in answer_text:
            parts = answer_text.split("Key takeaway:", 1)
            answer_text   = parts[0].strip()
            takeaway_html = f'<div class="takeaway">🔑 Key takeaway: {parts[1].strip()}</div>'

        sources_html = ""
        if msg.get("sources"):
            chips = "".join([f'<span class="source-chip">📜 {s[:40]}</span>' for s in msg["sources"]])
            sources_html = f'<div class="sources">{chips}</div>'

        st.markdown(f"""
<div class="msg-bot">
  <div class="avatar">🪷</div>
  <div>
    <div class="bubble">
      {answer_text}
      {takeaway_html}
    </div>
    {sources_html}
  </div>
</div>""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ── INPUT ─────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col_input, col_btn = st.columns([5, 1])

with col_input:
    user_input = st.text_input(
        label="question",
        label_visibility="collapsed",
        placeholder="Ask a spiritual question...",
        value=st.session_state.pending_question,
        key="user_input",
    )

with col_btn:
    ask_clicked = st.button("Ask 🙏", use_container_width=True)


# ── PROCESS QUESTION ─────────────────────────────────────
question = user_input.strip() or st.session_state.pending_question.strip()

if (ask_clicked or st.session_state.pending_question) and question:
    st.session_state.pending_question = ""
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.question_count += 1

    with st.spinner("Searching through discourses..."):
        try:
            answer, sources = query_rag(question, top_k=top_k)
            st.session_state.messages.append({
                "role":    "assistant",
                "content": answer,
                "sources": sources,
            })
        except Exception as e:
            st.session_state.messages.append({
                "role":    "assistant",
                "content": f"⚠️ Sorry, something went wrong: {str(e)}",
                "sources": [],
            })

    st.rerun()