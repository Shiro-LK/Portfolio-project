import pandas as pd
import streamlit as st
import requests, copy
from utils.request_fn import make_request, make_request_upload

 
    
def render_table(data_df):
    

    edited =  st.data_editor(data_df,
     
    #disabled=["delete"],
    key='edited_table',
    hide_index=True,
    )
    return edited 

def build_df(documents):
    data_df = pd.DataFrame(
    {
        "document": list(documents.keys()) ,
        "number":  list(documents.values()),
        "delete": [False] * len(list(documents.keys())),
    }
    )
    return data_df


def click_button():
    print(st.session_state.edited_data )
    for k, v in st.session_state.edited_table['edited_rows'].items():
        col = list(v.keys())[0]
        value = list(v.values())[0]
        st.session_state.edited_data.loc[k, col] = value
    # Collect the rows where the checkbox is selected
    selected_rows = st.session_state.edited_data[st.session_state.edited_data['delete'] == True]
    
    # Display the selected documents
    #st.write("Selected documents for deletion:", selected_rows['document'].tolist())
    print(st.session_state.edited_data['delete'])
    print('list: ', selected_rows['document'].tolist())
    documents_names = selected_rows['document'].tolist()
    documents_names = make_request(documents_names, mode="delete")
    
    documents_names = set(make_request( mode="listing").keys())
    print('left:', documents_names)
    # Remove the selected rows from the DataFrame
    st.session_state.edited_data = st.session_state.edited_data[st.session_state.edited_data['document'].apply(lambda x: x in  documents_names)].reset_index(drop=True)
    print(st.session_state.edited_data )
    print('----')
    # Update the table
    #st.experimental_rerun()

    
# get list of documents in database
 

if 'edited_data' not in st.session_state:
    docs =  make_request(  mode='listing')
    dataframe = build_df(docs)
    st.session_state.edited_data = dataframe.copy()
    #render_table(dataframe)
if 'edited_table' in st.session_state:
    print(st.session_state.edited_table)
# Store the edited data in a session state
# st.session_state.edited_data = 
render_table(st.session_state.edited_data)
#st.write(st.session_state.edited_data)
st.button('Submit', on_click=click_button)
