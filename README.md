# NVIDIA-NIM-Budget-2025QnA
# 💬 NVIDIA NIM Demo - Budget 2025 Q&A

This Streamlit web app allows users to **ask natural language questions** about India's **Budget 2025** and get accurate answers.  
It uses **NVIDIA NIM (NVIDIA Inference Microservices)** for running **LLaMA 3** models and **FAISS vector store** for semantic search over budget documents.

---

## 📌 Features
- **Ask questions** about the 2025 budget in plain English
- **Retrieval-Augmented Generation (RAG)** – answers are generated **only** from the provided context
- **FAISS vector store** for storing and retrieving embeddings
- **NVIDIA NIM LLaMA 3 model** for generating answers
- Shows **relevant document chunks** that contributed to the answer
- Tracks **response time**

---

## 🛠 Tech Stack
- **Frontend/UI**: [Streamlit](https://streamlit.io/)
- **LLM Provider**: [NVIDIA NIM](https://developer.nvidia.com/nim)
- **Embedding Model**: NVIDIA Embeddings
- **Vector Database**: FAISS (local)
- **LangChain**: For building retrieval and document chains
- **Python**: 3.10+

---

## 📂 Project Structure
├── app.py # Main Streamlit app
├── faiss_vector_store/ # Pre-built FAISS index for budget documents
├── requirements.txt # Python dependencies
├── .env # Environment variables (NVIDIA API key)
└── README.md # Project documentation
