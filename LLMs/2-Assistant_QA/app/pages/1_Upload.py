import streamlit as st
import os 
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import sqlite3 
import numpy as np 
import requests, copy
from utils.request_fn import make_request, make_request_upload


st.set_page_config(page_title="Upload PDFs", page_icon="📈")

st.markdown("# Upload PDFs")
st.sidebar.header("Upload PDFs")
st.write(
    """Upload PDFs which will be stored in our database.
    
    """
)



# Define the database path and the file path you want to add
 
chunk_size = 128
chunk_overlap = 0.5

st.session_state["counter"] = st.session_state.get("counter", 0)



def click_button():
    global pbar_v
    st.session_state["counter"] = 0.
    if uploaded_files:
        N = len(uploaded_files)
        count = np. linspace(0, 1.0, N+1).tolist()[1:]
        print(N)
        print(count)
        already_exists = set(make_request( to_remove=[], mode='listing').keys())
        for i, uploaded_file in enumerate(uploaded_files):
            
            if uploaded_file.name in already_exists:
                st.write(f'{uploaded_file.name} already exists in database.')
                st.session_state["counter"] = count[i]
                continue
            print(uploaded_file.name)

            
            success = make_request_upload( binary_data=uploaded_file.read(), filename=uploaded_file.name)
            
            if not success:
                st.write(f'{uploaded_file.name} uploading failed.')
            
            
            st.session_state["counter"] = count[i]
             
            st.session_state['progress_bar'].progress(st.session_state["counter"], text="Processing content from PDFs")


# Upload PDFs

uploaded_files = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)
#progress_bar = st.progress(pbar_v , text="Extracting content from PDFs")
st.session_state['progress_bar'] = st.sidebar.progress(st.session_state["counter"], text="Processing content from PDFs")



st.button('Submit', on_click=click_button)



