import streamlit as st
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import os

# 1. Force strict wide layout and load custom page configuration
st.set_page_config(
    page_title="AuraSearch PRO | Enterprise RAG Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Secure Token Entry
# Safe dynamic token loader
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception:
    HF_TOKEN = "YOUR_HUGGING_FACE_TOKEN_HERE"

# 3. Premium CSS Overrides (Glassmorphism, Neon Highlights, Custom Typography)
st.markdown("""
    <style>
    /* Import modern, clean tech font */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Apply base typography and background */
    html, body, [class*="st-emotion-cache"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background-color: #0B0F19 !important;
        color: #E2E8F0 !important;
    }
    
    /* Remove default Streamlit header bar decoration */
    header, [data-testid="stHeader"] {
        background-color: rgba(11, 15, 25, 0.8) !important;
        backdrop-filter: blur(12px) !important;
    }
    
    /* Sidebar Styling Override */
    section[data-testid="stSidebar"] {
        background-color: #0F1524 !important;
        border-right: 1px solid #1E293B !important;
    }
    
    /* Sleek Title Header block */
    .hero-container {
        padding: 40px 0px 20px 0px;
        text-align: left;
    }
    .hero-title {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #FFF 30%, #6366F1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
        margin-bottom: 5px;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 25px;
    }
    
    /* Custom CSS Search Button and Bar */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        padding: 12px 28px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
    }
    
    /* Premium Glassmorphic AI RAG Box */
    .rag-box {
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.6) 0%, rgba(49, 16, 66, 0.6) 100%) !important;
        backdrop-filter: blur(16px);
        border: 1.5px solid rgba(99, 102, 241, 0.4) !important;
        padding: 30px !important;
        border-radius: 16px !important;
        margin-bottom: 35px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        position: relative;
        overflow: hidden;
    }
    .rag-box::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #6366F1, #EC4899);
    }
    .rag-header {
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        color: #F8FAFC !important;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 15px;
    }
    .rag-content {
        color: #E2E8F0 !important;
        line-height: 1.7 !important;
        font-size: 1.05rem !important;
    }

    /* Highly polished Glassmorphic Paper Cards */
    .paper-card {
        background: rgba(30, 41, 59, 0.45) !important;
        backdrop-filter: blur(10px);
        padding: 28px !important;
        border-radius: 14px !important;
        margin-bottom: 22px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .paper-card:hover {
        transform: translateY(-4px) !important;
        background: rgba(30, 41, 59, 0.65) !important;
        border: 1px solid rgba(99, 102, 241, 0.25) !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25) !important;
    }
    .paper-title {
        color: #FFFFFF !important;
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        line-height: 1.4;
        margin-bottom: 12px;
    }
    .paper-abstract {
        color: #94A3B8 !important;
        font-size: 0.98rem !important;
        line-height: 1.6 !important;
    }
    
    /* Tech Badge for Retrieval Scores */
    .score-badge {
        display: inline-flex !important;
        align-items: center;
        background: rgba(16, 185, 129, 0.1) !important;
        color: #34D399 !important;
        border: 1px solid rgba(52, 211, 153, 0.2) !important;
        padding: 5px 12px !important;
        border-radius: 20px !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        margin-top: 15px !important;
        letter-spacing: 0.5px;
    }
    </style>
""", unsafe_allow_html=True)

# 4. Interactive Sidebar (Makes it feel like a real tool)
with st.sidebar:
    st.markdown("<h2 style='font-weight:800; color:#FFF; margin-bottom:5px;'>⚙️ Control Center</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B; font-size:0.85rem; margin-bottom:25px;'>System configuration and models</p>", unsafe_allow_html=True)
    
    st.markdown("### 🤖 Synthesis LLM")
    st.info("Llama-3.1-8B-Instruct (Active)")
    
    st.markdown("### 🗄️ Vector Database")
    st.success("ChromaDB Cluster Online")
    
    st.markdown("### 🛠️ Search Tuning")
    top_k = st.slider("Top Documents (K-Matches)", min_value=2, max_value=5, value=3)
    
    st.markdown("---")
    st.markdown("<p style='color:#475569; font-size:0.75rem;'>AuraSearch Pro v2.1 • Created for Professional Technical Portfolio</p>", unsafe_allow_html=True)

# 5. Hero UI Section
st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">⚡ AuraSearch PRO</h1>
        <p class="hero-subtitle">Next-generation RAG Engine powered by Local Vector Embeddings & Llama 3.1 Synthesis</p>
    </div>
""", unsafe_allow_html=True)

# 6. Database Loading & System Init
@st.cache_resource
def initialize_system():
    status = st.info("⏳ Initializing Local Vector Models... (Takes 30s on cold boot)")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    status.empty()
    
    chroma_client = chromadb.PersistentClient(path="./chroma_storage")
    collection = chroma_client.get_or_create_collection(name="arxiv_production")
    
    if not os.path.exists("arxiv_sample.csv"):
        st.error("Error: 'arxiv_sample.csv' missing from project directory.")
        st.stop()
        
    df = pd.read_csv("arxiv_sample.csv")
    
    if collection.count() == 0:
        status = st.warning("⏳ Deep indexing 2,000 reference abstracts. Preparing cluster...")
        documents = df['abstract'].fillna("").tolist()
        metadatas = df[['title', 'id']].to_dict(orient='records')
        ids = df['id'].astype(str).tolist()
        
        embeddings = model.encode(documents, show_progress_bar=True).tolist()
        collection.add(embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids)
        status.success("✅ Neural search index fully compiled!")
        
    return collection, model

try:
    collection, model = initialize_system()
except Exception as e:
    st.error(f"Failed to bootstrap database cluster: {e}")
    st.stop()

# 7. Helper function for RAG Generation
def generate_summary(query, papers):
    if HF_TOKEN == "YOUR_HUGGING_FACE_TOKEN_HERE" or HF_TOKEN == "":
        return "⚠️ Complete the token variable inside your Python code to authorize the LLM engine."
        
    api_url = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    
    context = "\n\n".join([f"Title: {p['title']}\nAbstract: {p['abstract']}" for p in papers])
    
    payload = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [
            {
                "role": "system",
                "content": "You are an AI research assistant. Synthesize a highly technical, concise, 3-bullet-point summary explaining the core findings of the provided papers as they relate to the user's query. Do not write intros, outros, or meta-commentary."
            },
            {
                "role": "user",
                "content": f"User Query: {query}\n\nPapers:\n{context}"
            }
        ],
        "max_tokens": 250
    }
    
    response = requests.post(api_url, headers=headers, json=payload, timeout=10)
    
    try:
        result = response.json()
        if "choices" in result:
            return result["choices"][0]["message"]["content"]
        elif "error" in result:
            return f"HuggingFace Router Error: {result['error']}"
        return f"Unexpected API Response Structure: {result}"
    except Exception as e:
        return f"Could not generate neural summary: {e}"

# 8. User Input Bar Layout
col1, col2 = st.columns([6, 1])
with col1:
    user_query = st.text_input("Query", placeholder="Search topics (e.g., 'object recognition in cameras', 'LLM reasoning bounds')...", label_visibility="collapsed")
with col2:
    search_button = st.button("Query Engine", use_container_width=True)

# 9. Main Processing Thread
if user_query or search_button:
    if user_query.strip() == "":
        st.warning("Please enter a conceptual query first.")
    else:
        # Step A: Local Semantic Search
        query_vector = model.encode([user_query]).tolist()
        results = collection.query(query_embeddings=query_vector, n_results=top_k)
        
        matched_papers = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                matched_papers.append({
                    "title": results['metadatas'][0][i]['title'],
                    "abstract": results['documents'][0][i],
                    "distance": results['distances'][0][i]
                })
        
        if matched_papers:
            # Step B: Display RAG Synthesis
            st.markdown("<h3 style='margin-top: 30px; font-weight:700; color:#F1F5F9;'>🤖 AI Abstract Synthesis (RAG)</h3>", unsafe_allow_html=True)
            
            with st.spinner("Streaming summary tokens from Meta Llama 3.1..."):
                try:
                    summary = generate_summary(user_query, matched_papers)
                    if "HuggingFace Router Error" in summary or "Unexpected" in summary:
                        st.error(summary)
                    else:
                        st.markdown(f"""
                            <div class="rag-box">
                                <div class="rag-header">
                                    <span>🧠</span> Contextual Insight Synthesis
                                </div>
                                <div class="rag-content">
                                    {summary}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                except requests.exceptions.ConnectionError:
                    st.warning("📡 **Offline Mode Activated**: Host network is blocking the external LLM router. Presenting retrieved papers directly from local index below.")
                except Exception as e:
                    st.error(f"Synthesizer encountered a network error: {e}")
            
            # Step C: List the Papers
            st.markdown("<h3 style='margin-top:20px; font-weight:700; color:#F1F5F9;'>📂 Retrieved Reference Source Papers</h3>", unsafe_allow_html=True)
            st.markdown("<hr style='border-color: #1E293B; margin-bottom: 25px;' />", unsafe_allow_html=True)
            
            for paper in matched_papers:
                confidence = int(100 / (1 + paper['distance']))
                st.markdown(f"""
                    <div class="paper-card">
                        <div class="paper-title">{paper['title']}</div>
                        <div class="paper-abstract">{paper['abstract']}</div>
                        <div class="score-badge">🔍 Match Confidence: {confidence}%</div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No matching abstracts found inside the vector space database.")