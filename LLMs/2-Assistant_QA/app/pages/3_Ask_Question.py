import streamlit as st
import fitz  # PyMuPDF
import requests, base64, numpy as np
from streamlit_pdf_viewer import pdf_viewer
from utils.request_fn import make_request, make_request_upload, make_request_query


def generate_blank_pdf():
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    import io

    # Create a BytesIO buffer
    buffer = io.BytesIO()

    # Create a canvas object with the buffer
    c = canvas.Canvas(buffer, pagesize=letter)

    # Draw a blank page
    c.showPage()

    # Save the PDF content to the buffer
    c.save()

    # Get the PDF content as a byte object
    pdf_bytes = buffer.getvalue()

    # Close the buffer
    buffer.close()

    print(f"PDF generated with {len(pdf_bytes)} bytes.")
    return pdf_bytes


def displayPDF(uploaded_file):
    if type(uploaded_file) == str:
        uploaded_file = open(uploaded_file, 'rb').read()
    
    # Read file as bytes:
    bytes_data = uploaded_file

    # Convert to utf-8
    base64_pdf = base64.b64encode(bytes_data).decode('utf-8')

    # Embed PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'

    # Display file
    st.markdown(pdf_display, unsafe_allow_html=True)
    
def click_button():
     
    query = st.session_state['question']
    print('Query:', query)
    r = make_request_query(query)
    print(r)
    st.session_state['response']= r #st.text_area(r, height=200)
    st.session_state['pdf'] = "../tmp/temp_highlighted.pdf"
    #displayPDF(st.session_state['pdf'])
    
    
# Path to the PDF file
st.set_page_config(
    page_title="Q&A", layout="wide"
    #page_icon="👋",
)

chat, viewer = st.columns([50, 50])

if 'pdf' not in st.session_state:
    st.session_state['pdf'] = generate_blank_pdf()
if 'response' not in st.session_state:
    st.session_state['response'] = ''


with chat:
    #if 'question' not in st.session_state:
    st.session_state['question'] = st.text_input("Write your question here:") 
    response = st.text_area('Answer', st.session_state['response'], height=200)
    st.button('Submit', on_click=click_button)
    
# Column 2: PDF Viewer
with viewer:
    displayPDF(st.session_state['pdf'])
    #pdf_viewer(input=st.session_state['pdf'],
    #               width=700)