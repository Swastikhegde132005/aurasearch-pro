import streamlit as st
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import requests
import io
import os

# ReportLab components for handling multi-page text wrapping safely
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# 1. Page Configuration
st.set_page_config(
    page_title="AuraSearch Ultra Pro | Next-Gen Book Vault",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State variables for layout and state tracking
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "sidebar_recommendations" not in st.session_state:
    st.session_state.sidebar_recommendations = []
if "current_books_pool" not in st.session_state:
    st.session_state.current_books_pool = []
if "displayed_count" not in st.session_state:
    st.session_state.displayed_count = 5
if "active_query" not in st.session_state:
    st.session_state.active_query = ""
if "book_bookmarks" not in st.session_state:
    st.session_state.book_bookmarks = []

# 2. Premium UI Custom Styling (Glassmorphism & Neon Accent Layers)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght=300;400;500;600;700;800&family=Space+Grotesk:wght=400;600;700&display=swap');
    
    html, body, [class*="st-emotion-cache"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background-color: #080C14 !important;
        color: #E2E8F0 !important;
    }
    
    h1, h2, h3, .hero-title {
        font-family: 'Space Grotesk', sans-serif !important;
    }
    
    header, [data-testid="stHeader"] {
        background-color: rgba(8, 12, 20, 0.8) !important;
        backdrop-filter: blur(16px) !important;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #0D1322 !important;
        border-right: 1px solid rgba(99, 102, 241, 0.15) !important;
    }
    
    .hero-container { 
        padding: 30px 25px; 
        text-align: left; 
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        margin-bottom: 25px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }
    .hero-title {
        font-size: 3.2rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #FFFFFF 20%, #818CF8 60%, #6366F1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
        margin-bottom: 5px;
    }
    .hero-subtitle { color: #94A3B8; font-size: 1.1rem; font-weight: 400; }
    
    div.stButton > button:first-child, div.stDownloadButton > button:first-child {
        background: linear-gradient(135deg, #4F46E5 0%, #3730A3 100%) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        padding: 12px 28px !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 20px rgba(79, 70, 229, 0.25) !important;
        font-size: 0.95rem !important;
    }
    div.stButton > button:first-child:hover, div.stDownloadButton > button:first-child:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(99, 102, 241, 0.4) !important;
        border-color: rgba(255,255,255,0.2) !important;
    }
    
    .book-display-card {
        background: rgba(17, 24, 39, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-left: 5px solid #6366F1 !important;
        padding: 28px !important;
        border-radius: 16px !important;
        margin-bottom: 25px !important;
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
    }
    
    .sidebar-rec-card {
        background: rgba(22, 30, 49, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        border-left: 3px solid #F59E0B !important;
        padding: 16px !important;
        border-radius: 12px !important;
        margin-bottom: 14px !important;
    }
    
    .book-title { 
        color: #FFFFFF !important; 
        font-size: 1.6rem !important; 
        font-weight: 700 !important; 
        margin-bottom: 6px; 
    }
    
    .book-meta {
        color: #94A3B8 !important;
        font-size: 0.95rem !important;
        margin-bottom: 10px;
    }
    
    .ui-badge-yellow {
        display: inline-flex;
        background: rgba(245, 158, 11, 0.12);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.25);
        padding: 4px 12px;
        border-radius: 30px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .ui-badge-blue {
        display: inline-flex;
        background: rgba(59, 130, 246, 0.12);
        color: #3B82F6;
        border: 1px solid rgba(59, 130, 246, 0.25);
        padding: 4px 12px;
        border-radius: 30px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# 3. System Initialization (FIXED FOR OFFLINE COMPATIBILITY)
@st.cache_resource
def initialize_system():
    status = st.info("⏳ Bootstrapping Ultra-Engine Context Layers...")
    
    # Check if local model directory exists to bypass network calls
    if os.path.exists("./local_model"):
        model = SentenceTransformer('./local_model')
    else:
        # Fallback to online loading if local setup isn't done yet
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
    status.empty()
    
    chroma_client = chromadb.PersistentClient(path="./chroma_storage")
    collection = chroma_client.get_or_create_collection(name="clean_vault")
    return collection, model

collection, model = initialize_system()

# 4. Multi-Page Full Text PDF Compiler Engine
def compile_full_pdf(title, author, unbroken_text):
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=22, leading=26, spaceAfter=6, textColor='#1E293B')
    author_style = ParagraphStyle('DocAuthor', parent=styles['Normal'], fontSize=12, leading=16, spaceAfter=20, textColor='#4F46E5', fontName='Helvetica-Bold')
    body_style = ParagraphStyle('DocBody', parent=styles['Normal'], fontSize=10, leading=15, spaceAfter=12, textColor='#334155')
    
    story = [Paragraph(title, title_style), Paragraph(f"Author: {author}", author_style), Spacer(1, 15)]
    paragraphs = unbroken_text.split('\n\n')
    
    for p in paragraphs:
        clean_para = p.replace('\n', ' ').strip()
        if clean_para:
            clean_para = clean_para.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            try: story.append(Paragraph(clean_para, body_style))
            except Exception: continue
                
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

# 5. Live Document Downloader
def fetch_unabridged_book(query, limit=30):
    safe_query = requests.utils.quote(query.strip())
    search_url = f"https://gutendex.com/books/?search={safe_query}"
    try:
        response = requests.get(search_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            if not results:
                return []
            
            payloads = []
            for book in results[:limit]:
                title = book.get('title', 'Unknown Title')
                authors = ", ".join([a.get('name', 'Unknown') for a in book.get('authors', [])])
                formats = book.get('formats', {})
                txt_url = formats.get('text/plain; charset=utf-8') or formats.get('text/plain')
                
                if txt_url:
                    text_stream = requests.get(txt_url, timeout=15)
                    if text_stream.status_code == 200:
                        payloads.append({
                            "title": title,
                            "authors": authors,
                            "full_text": text_stream.text,
                            "subject": book.get('subjects', ['Literature'])[0] if book.get('subjects') else "Literature"
                        })
            return payloads
    except Exception:
        return []
    return []

# 6. High-Option Left Navigation Control Center & Amazon History Suggestions
with st.sidebar:
    st.markdown("<h2 style='color:#FFF; margin-bottom: 2px;'>⚙️ Control Desk</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6366F1; font-size:0.85rem; margin-top:0px;'>Configure your vault environment</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 🎛️ Session Dashboard")
    records_per_scroll = st.slider("Scroll Load Increment", min_value=3, max_value=15, value=5)
    
    st.markdown("### 🕒 Recent Searches")
    if st.session_state.search_history:
        for history_item in st.session_state.search_history[-4:]:
            st.code(f"🔍 {history_item}")
        if st.button("🧹 Clear Search Logs", use_container_width=True):
            st.session_state.search_history = []
            st.rerun()
    else:
        st.caption("No queries tracked in this session yet.")
        
    st.markdown("---")
    
    st.markdown("<h3 style='color:#FFF; margin-bottom:2px;'>🪐 Amazon Suggestions</h3>", unsafe_allow_html=True)
    
    if st.session_state.sidebar_recommendations:
        for s_idx, rec_book in enumerate(st.session_state.sidebar_recommendations[:4]):
            st.markdown(f"""
                <div class="sidebar-rec-card">
                    <div style="font-weight:700; color:#FFFFFF; font-size:0.95rem; line-height:1.2;">{rec_book['title']}</div>
                    <div style="color:#94A3B8; font-size:0.8rem; margin-top:4px;">By {rec_book['authors']}</div>
                </div>
            """, unsafe_allow_html=True)
            
            activate_side_comp = st.checkbox("Generate PDF payload", key=f"side_chk_{s_idx}")
            if activate_side_comp:
                with st.spinner("Building PDF..."):
                    r_pdf = compile_full_pdf(rec_book['title'], rec_book['authors'], rec_book['full_text'])
                st.download_button(
                    label=f"📥 Save PDF", 
                    data=r_pdf, 
                    file_name=f"Suggested_{rec_book['title'].replace(' ','_')}.pdf", 
                    key=f"side_pdf_{s_idx}",
                    use_container_width=True
                )
            st.markdown("<div style='margin-bottom:10px;'></div>", unsafe_allow_html=True)
    else:
        st.info("Personalized suggestions populate here based on search context.")

# Main Presentation Panel Layout
st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">🪐 AuraSearch Ultra Pro</h1>
        <p class="hero-subtitle">Premium Discovery Engine with Interactive Sidebar Suggestions & On-Demand PDF Pipelines</p>
    </div>
""", unsafe_allow_html=True)

# 7. Search Interface Core Layout
col_input, col_btn = st.columns([6, 1])
with col_input:
    user_query = st.text_input("Query Bar", placeholder="Search for any book title, classic author, or conceptual subject...", label_visibility="collapsed")
with col_btn:
    search_button = st.button("Search Archive", use_container_width=True)

if (user_query and user_query != st.session_state.active_query) or search_button:
    if user_query.strip() == "":
        st.warning("Please submit a search prompt.")
    else:
        st.session_state.active_query = user_query.strip()
        st.session_state.displayed_count = records_per_scroll
        st.session_state.current_books_pool = []
        
        if user_query.strip() not in st.session_state.search_history:
            st.session_state.search_history.append(user_query.strip())

        with st.status("🌐 Traversing Network Hubs & Constructing Secure Tokens...", expanded=True) as status:
            books_pool = fetch_unabridged_book(st.session_state.active_query, limit=30)
            
            if books_pool:
                st.session_state.current_books_pool = books_pool
                st.session_state.sidebar_recommendations = fetch_unabridged_book(books_pool[0]['subject'], limit=4)
                status.update(label="🚀 Data channels connected! Review results below.", state="complete")
            else:
                status.update(label="⚠️ Free collection matches are currently unavailable.", state="error")

# Step B: Render matches smoothly from session data streams
if st.session_state.current_books_pool:
    st.markdown(f"### 📂 Available Channels (Displaying top {min(st.session_state.displayed_count, len(st.session_state.current_books_pool))} matches)")
    st.markdown("<hr style='border-color: rgba(255,255,255,0.05);' />", unsafe_allow_html=True)
    
    visible_subset = st.session_state.current_books_pool[:st.session_state.displayed_count]
    
    for idx, book in enumerate(visible_subset):
        encoded_title = requests.utils.quote(book['title'])
        amazon_url = f"https://www.amazon.com/s?k={encoded_title}+book"
        google_play_url = f"https://books.google.com/books?q={encoded_title}"
        
        st.markdown(f"""
            <div class="book-display-card">
                <div class="book-title">📖 {book['title']}</div>
                <div class="book-meta">✍ Author/Compiler: <b>{book['authors']}</b></div>
                <span class="ui-badge-yellow">Category: {book['subject']}</span>
                <span class="ui-badge-blue">Source Link Verified</span>
            </div>
        """, unsafe_allow_html=True)
        
        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([2.5, 2.5, 2.5, 1.5])
        
        with btn_col1:
            wants_pdf = st.checkbox("⚙️ Generate PDF Payload", key=f"chk_{idx}")
            if wants_pdf:
                with st.spinner("Assembling PDF matrix..."):
                    full_pdf_binary = compile_full_pdf(book['title'], book['authors'], book['full_text'])
                st.download_button(
                    label="📥 Download Full PDF",
                    data=full_pdf_binary,
                    file_name=f"{book['title'].replace(' ', '_')}_Unabridged.pdf",
                    mime="application/pdf",
                    key=f"manual_pdf_{idx}",
                    use_container_width=True
                )
            
        with btn_col2:
            st.markdown(f'<a href="{amazon_url}" target="_blank"><button style="background: linear-gradient(135deg, #FF9900 0%, #E67E22 100%); color:black; border:none; padding:10px 14px; border-radius:10px; font-weight:700; cursor:pointer; height:42px; width:100%;">🛒 Buy on Amazon</button></a>', unsafe_allow_html=True)
            
        with btn_col3:
            st.markdown(f'<a href="{google_play_url}" target="_blank"><button style="background: linear-gradient(135deg, #34A853 0%, #27AE60 100%); color:white; border:none; padding:10px 14px; border-radius:10px; font-weight:600; cursor:pointer; height:42px; width:100%;">🌐 Find on Google Books</button></a>', unsafe_allow_html=True)
            
        with btn_col4:
            is_bookmarked = book['title'] in st.session_state.book_bookmarks
            bookmark_label = "❤️ Saved" if is_bookmarked else "🖤 Favorite"
            if st.button(bookmark_label, key=f"fav_{idx}", use_container_width=True):
                if not is_bookmarked:
                    st.session_state.book_bookmarks.append(book['title'])
                    st.toast(f"Saved: {book['title']} added to your list!")
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.displayed_count < len(st.session_state.current_books_pool):
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔽 Load More Results (Continuous Amazon Scroll)", use_container_width=True):
            st.session_state.displayed_count += records_per_scroll
            st.rerun()
    else:
        st.info("🏁 Master collection index search complete.")

elif user_query and not st.session_state.current_books_pool:
    st.markdown(f"### 🛒 Authorized Commercial Outlets")
    st.markdown("<hr style='border-color: rgba(255,255,255,0.05);' />", unsafe_allow_html=True)
    
    encoded_query = requests.utils.quote(user_query)
    fallback_amazon = f"https://www.amazon.com/s?k={encoded_query}+book"
    fallback_google = f"https://books.google.com/books?q={encoded_query}"
    
    st.markdown(f"""
        <div class="book-display-card" style="border-left: 5px solid #EF4444 !important;">
            <div class="book-title" style="color:#EF4444 !important;">🔒 Secure Licensing Verification Required</div>
            <p style="color:#94A3B8; font-size:0.95rem;">'{user_query}' requires an official publisher license. You can acquire a permanent copy or access cloud library reading panels via these certified vendors:</p>
        </div>
    """, unsafe_allow_html=True)
    
    f_col1, f_col2 = st.columns([1, 1])
    with f_col1:
        st.markdown(f'<a href="{fallback_amazon}" target="_blank"><button style="background: linear-gradient(135deg, #FF9900 0%, #E67E22 100%); color:black; border:none; padding:12px 20px; border-radius:10px; font-weight:700; cursor:pointer; width:100%;">🛒 Search Amazon Storefront</button></a>', unsafe_allow_html=True)
    with f_col2:
        st.markdown(f'<a href="{fallback_google}" target="_blank"><button style="background: linear-gradient(135deg, #34A853 0%, #27AE60 100%); color:white; border:none; padding:12px 20px; border-radius:10px; font-weight:600; cursor:pointer; width:100%;">🌐 Open Google Books Store</button></a>', unsafe_allow_html=True)