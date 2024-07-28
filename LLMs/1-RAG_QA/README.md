# PDF Q&A with RAG (Retrieval Augmented Generation)


This project uses Retrieval Augmented Generation (RAG) to answer questions from a PDF. The process involves several steps:

1. Upload a PDF: Start by uploading your PDF document.
2. OCR and Chunking: Apply Optical Character Recognition (OCR) with Doctr, and then chunk the text using Llama-index.
3. Embedding Extraction: Extract embeddings for each chunk using a Huggingface open-source model.
4. Retrieval: Perform the retrieval step using Llama-index.
5. Chunk Selection and Merging: Select the top 15 chunks and merge any overlapping chunks. These merged chunks form the search results.
6. Answer Generation: Use the Llama 3.1 8b model to generate an answer based on the search results.
7. Optional: View the PDF with highlighted search results in Gradio. The relevant paragraphs will be highlighted in yellow.



![image](https://github.com/Shiro-LK/Portfolio-project/tree/main/LLMs/1-RAG_QA/images/gradio_app.png)