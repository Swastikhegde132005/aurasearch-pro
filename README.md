# 🪐 AuraSearch Ultra Pro

A premium, next-generation book discovery engine and on-demand PDF compilation tool built with Streamlit. The system leverages semantic vector indexes and classic search algorithms to find public-domain books, stream them line-by-line, and selectively compile them into clean, multi-page print documents.

---

## 🚀 Key Features

* **Premium Glassmorphic UI:** A dark-themed marketplace layout utilizing neon accent layers, active state custom badges, and responsive view containers.
* **On-Demand PDF Compilation:** High-volume ReportLab rendering pipelines that only activate when explicitly selected via user checkboxes, optimizing local hardware performance.
* **Amazon-Style History Suggestions:** An active recommendation matrix living permanently inside the left sidebar that surfaces tailored literary suggestions dynamically mapped to your active session history.
* **Continuous Lazy Loading:** An infinite-scroll simulation loop letting you incrementally pull deeper pools of matching search engine index paths without resetting page states.
* **Commercial Marketplace Fallbacks:** Clean routing mechanisms that auto-generate direct purchase links to official **Amazon** and **Google Books** store catalogs for modern copyrighted items.

---

## 🛠️ Project Architecture

* **Frontend Dashboard:** Streamlit UI Component Framework
* **Vector Embeddings Matrix:** `SentenceTransformer` text-embeddings mapping
* **Vector Index Store:** `ChromaDB` localized vector database management
* **Lexical Ranking Pass:** `BM25Okapi` search indexing arrays
* **Document Compilation Platform:** `ReportLab` multi-page PDF generation library

---

## 💻 Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
   cd YOUR_REPOSITORY_NAME