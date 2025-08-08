# ------------------------------
# 1. Import Required Libraries
# ------------------------------
import streamlit as st
import os
import time
from dotenv import load_dotenv

# LangChain & NVIDIA NIM (NVIDIA Inference Microservices) components
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

# ------------------------------
# 2. Load Environment Variables
# ------------------------------
load_dotenv()  # Reads values from .env file
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")  # Set NVIDIA API Key

# ------------------------------
# 3. Load Pre-built Vector Store
# ------------------------------
# Using NVIDIA embeddings to ensure compatibility with the vector store
embeddings = NVIDIAEmbeddings()

# Load an existing FAISS vector store (previously created and saved)
vectorstore = FAISS.load_local(
    "faiss_vector_store",  # Path to saved FAISS index
    embeddings,
    allow_dangerous_deserialization=True  # Required for loading
)

# Store vectorstore in session state for reusability
st.session_state.vectorstore = vectorstore

# ------------------------------
# 4. Streamlit UI Title
# ------------------------------
st.title("💬 NVIDIA NIM Demo - Budget 2025 Q&A")

# ------------------------------
# 5. Initialize LLM Model
# ------------------------------
# Using NVIDIA NIM LLaMA 3 model (70B parameter instruction-tuned)
llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct")

# ------------------------------
# 6. Define Prompt Template
# ------------------------------
# This template ensures the model only uses given context to answer
prompt_template = ChatPromptTemplate.from_template("""
Answer the question **only** based on the provided context.
Be as accurate and clear as possible.

<context>
{context}
</context>

Question: {input}
""")

# ------------------------------
# 7. Get User Question
# ------------------------------
user_question = st.text_input("Enter your question about Budget 2025:")

# ------------------------------
# 8. Processing User Query
# ------------------------------
if user_question:
    # Step 8.1: Create a document chain (LLM + prompt)
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    
    # Step 8.2: Create a retriever from the vectorstore
    retriever = vectorstore.as_retriever()
    
    # Step 8.3: Combine retriever + document chain into one retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # Step 8.4: Track response time
    start = time.process_time()
    response = retrieval_chain.invoke({'input': user_question})
    elapsed_time = time.process_time() - start
    
    # Step 8.5: Display the answer
    st.subheader("Answer:")
    st.write(response['answer'])
    st.caption(f"⏱ Response time: {elapsed_time:.2f} seconds")
    
    # Step 8.6: Show relevant context chunks
    with st.expander("🔍 View Relevant Document Chunks"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("---")
