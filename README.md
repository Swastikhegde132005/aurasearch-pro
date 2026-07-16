# ⚡ AuraSearch PRO | Enterprise Semantic RAG Search Engine

AuraSearch PRO is a production-ready, high-performance Retrieval-Augmented Generation (RAG) search engine. Unlike standard keyword searches that match exact text strings, AuraSearch PRO uses local vector embeddings to understand the **conceptual intent** behind a user's query, mapping natural language to a database of 2,000 academic abstracts in real-time.

---

## 🚀 Key Features

*   **Local Semantic Search Engine:** Transforms unstructured search queries into high-dimensional vectors locally using the `all-MiniLM-L6-v2` SentenceTransformer model.
*   **Vector Database (ChromaDB):** Leverages a persistent local ChromaDB database on disk to bypass repetitive startup indexing and perform top-$k$ spatial vector queries in milliseconds.
*   **Contextual AI Synthesis (RAG):** Orchestrates Meta's **Llama 3.1 8B Instruct** via Hugging Face's stable router API to dynamically synthesize retrieved context into structured summaries.
*   **Dynamic Tuning Control Panel:** Built with an interactive sidebar that lets users dynamically adjust retrieval depth (top-$k$ matches) and monitor live system status.
*   **Graceful Offline Fallbacks:** Designed with robust connection exception-handling, enabling the search database to gracefully degrade to local-only search in network-restricted or offline environments.
*   **Premium Glassmorphic UI:** A custom-styled dark interface with responsive hover states, custom typography (`Plus Jakarta Sans`), and glowing layout containers that completely mask default Streamlit components.

---

## 🛠️ Tech Stack

*   **Backend & App Framework:** Python, Streamlit
*   **Vector Database:** ChromaDB (Persistent SQLite Client)
*   **Embedding Model:** SentenceTransformers (`all-MiniLM-L6-v2`)
*   **LLM Integration:** Hugging Face Stable Router API (Meta Llama 3.1 8B Instruct)
*   **Data Processing:** Pandas, NumPy
*   **Styling:** Advanced Custom CSS Injection (Glassmorphism design patterns)

---

## 📐 System Architecture & Data Flow

```text
  [ User Query ]  ──>  [ Local Embedding Model ]
                             │
                             ▼ (384-dimensional vector)
                      [ ChromaDB Vector Query ]
                             │
                             ▼ (Top-K Matches & Distances Calculated)
                      [ Context Assembled ] ──> [ Llama 3.1 Synthesis ]
                             │                            │
                             ▼ (Offline Render)           ▼ (Online Generation)
                      [ Paper Cards Displayed ]  <──  [ Glowing RAG Summary Box ]