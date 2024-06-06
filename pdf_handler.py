# import os
# import streamlit as st
# import pickle
# from embeddings import get_pdf_text, get_text_chunks, create_embeddings, store_vector


# def create_new_vector_db(pdf, file_name):
#     with st.spinner("Creating new vector data"):
#         text = get_pdf_text(pdf)
#         chunks = get_text_chunks(text)
#         vectordb = create_embeddings(chunks)
#         vectordb = store_vector(vectordb, file_name)
#     return vectordb


# def fetch_vector_db(file_name):
#     if os.path.exists(f"{file_name}.pkl"):
#         with open(f"{file_name}.pkl", "rb") as f:
#             with st.spinner("Fetching existing vector data"):
#                 vectordb = pickle.load(f)
#     return vectordb



# def handle_pdf_upload(pdf):
#     if pdf:
#         file_name = pdf.name[:-4]    #get filename without .pdf
#         if os.path.exists(f"{file_name}.pkl"):
#             st.write("PDF file already exists.")
#             with open(f"{file_name}.pkl", "rb") as f:
#                 with st.spinner("Fetching existing vector data"):
#                     vectordb = pickle.load(f)

#         else:
#             vectordb = create_new_vector_db(pdf, file_name)
#             st.write("Vector data created successfully.")

#         return vectordb

#     else:                             
#         return fetch_vector_db(file_name)


import os
import streamlit as st
import pickle
from embeddings import get_pdf_text, get_text_chunks, create_embeddings, store_vector

def create_new_vector_db(pdf, file_name):
    with st.spinner("Creating new vector data"):
        text = get_pdf_text(pdf)
        chunks = get_text_chunks(text)
        vectordb = create_embeddings(chunks)
        vectordb = store_vector(vectordb, file_name)
    return vectordb

def fetch_vector_db(file_name):
    file_path = os.path.join('vector_data', f"{file_name}.pkl")
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            with st.spinner("Fetching existing vector data"):
                vectordb = pickle.load(f)
    return vectordb

def handle_pdf_upload(pdf):
    if pdf:
        file_name = pdf.name[:-4]    # get filename without .pdf
        file_path = os.path.join('vector_data', f"{file_name}.pkl")
        if os.path.exists(file_path):
            st.write("PDF file already exists.")
            with open(file_path, "rb") as f:
                with st.spinner("Fetching existing vector data"):
                    vectordb = pickle.load(f)
        else:
            vectordb = create_new_vector_db(pdf, file_name)
            st.write("Vector data created successfully.")
        return vectordb
    else:
        return fetch_vector_db(file_name)
