import streamlit as st





st.set_page_config(
    page_title="Q&A Agent",
    page_icon="👋",
)

st.write("# Welcome! 👋")

#st.sidebar.success("Select a demo above.")

st.markdown(
    """
## Q&A Assistant with RAG and LLM Integration:

- Document Upload: Users can upload documents.
- Chunking and Storage: The documents are divided into manageable chunks and stored in a ChromaDB database.
- Query Answering: Questions are answered using Retrieval-Augmented Generation (RAG).
- Answer Source Display: The relevant PDF section containing the answer is displayed.

*Technologies Used: Langchain, Huggingface, Streamlit, FastAPI.*

Note: Llama3.1 is used, however for better performances, you can use OPENAI or CLAUDE LLMs.
"""
)
