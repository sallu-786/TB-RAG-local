import os
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI
from pdf_handler import handle_pdf_upload
from langchain_community.llms import CTransformers

import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def create_llm(model_path=config["model_path"]["model_local"], model_type=config["model_type"], model_config=config["model_config"]):
    llm = CTransformers(model=model_path, model_type=model_type, config=model_config)
    return llm


USER_NAME = "user"
ASSISTANT_NAME = "assistant"


st.title("TB PDF-Chatbot")  # Name of Application
st.write("Please upload your pdf file and type message")
                
def response_chatgpt(user_msg: str, input_documents, chat_history: list = []):
    system_msg = """You are an Assistant. Answer the questions based only on the provided documents. 
                    If the information is not in the documents or you cant find it, say to user you don't know the answer
                    but you can tell user the they may find something relevant in sources given below and then you display sources"""
    messages = [{"role": "assistant", "content": system_msg}]

    # If there is a chat log, add it to the messages list
    if len(chat_history) > 0:
        for chat in chat_history:
            messages.append({"role": chat["name"], "content": chat["msg"]})

    # Add user message to messages list
    messages.append({"role": USER_NAME, "content": user_msg})

    # Append input documents to the messages list
    for doc in input_documents:
        messages.append({"role": "user", "content": f"Document snippet:\n{doc['content']}"})

    try:
        model = create_llm()
        response = model.invoke(messages,temperature=0) 
        return {
            "answer": response,
            "sources": input_documents
        }
    except Exception as e:
        st.error(f"Could not load the model: {str(e)}")
        return None


def main():
    # Sidebar for PDF upload

    with st.sidebar:
        st.title('PDF Chat Loader')
        pdf = st.file_uploader("Upload your PDF", accept_multiple_files=False, type='pdf')
        send_button = st.button("Submit", key="send_button")
        if send_button:
            try:
                vectordb = handle_pdf_upload(pdf)
                pdf_name = pdf.name
            except Exception as e:
                st.error(f"Please submit a valid pdf file")
            if vectordb:
                st.session_state.vectordb = vectordb
                st.session_state.pdf_name = pdf_name

    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []

    user_msg = st.chat_input("Enter your message here")
    if user_msg:
        # Show previous chat logs
        for chat in st.session_state.chat_log:
            with st.chat_message(chat["name"]):
                st.write(chat["msg"])

        # Display the latest user messages
        with st.chat_message(USER_NAME):
            st.write(user_msg)

        try:
            docs = st.session_state.vectordb.similarity_search(query=user_msg, k=3)
            doc_texts = [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
        except Exception as e:
            st.error(f"Vector database not found: {str(e)}")

        # Get the response from GPT
        with st.spinner("Loading answer..."):
            response = response_chatgpt(user_msg, doc_texts, chat_history=st.session_state.chat_log)
        if response:
            with st.chat_message(ASSISTANT_NAME):
                assistant_msg = response["answer"]
                assistant_response_area = st.empty()
                assistant_response_area.write(assistant_msg)

                # Display the first source document
                invalid_responses = ["I don't know", "I couldn't find", "there is no information about", "I'm sorry", "is not mentioned", "I don't have information", "there is no specific information", "there is no mention"]
                if not any(response in assistant_msg for response in invalid_responses):
                    st.write("### Source")
                    pdf_name = st.session_state.pdf_name if "pdf_name" in st.session_state else "Source unavailable"
                    if response["sources"]:
                            source = response["sources"][0]
                        # for index, source in enumerate(response["sources"], start=1):
                            page_number = source["metadata"].get('page')
                            if page_number:
                                st.write(f"- File: {pdf_name} ----- Page Number: {page_number}")
                            else:
                                st.write(f"- File: {pdf_name} ----- Page Number: Unavailable")

                

            # Add chat logs to the session
            st.session_state.chat_log.append({"name": USER_NAME, "msg": user_msg})
            st.session_state.chat_log.append({"name": ASSISTANT_NAME, "msg": assistant_msg})

if __name__ == "__main__":
    main()




# def response_chatgpt(user_msg: str, input_documents, chat_history: list = []):
#     system_msg = """You are an Assistant. Answer the questions based only on the provided documents. 
#                     If the information is not in the documents or you cant find it, say to user you don't know the answer
#                     but you can tell user the they may find something relevant in sources given below and then you display sources"""
#     messages = [{"role": "system", "content": system_msg}]

#     # If there is a chat log, add it to the messages list
#     if len(chat_history) > 0:
#         for chat in chat_history:
#             messages.append({"role": chat["name"], "content": chat["msg"]})

#     # Add user message to messages list
#     messages.append({"role": USER_NAME, "content": user_msg})

#     # Append input documents to the messages list
#     for doc in input_documents:
#         messages.append({"role": "user", "content": f"Document snippet:\n{doc['content']}"})

#     try:
#         response = client.chat.completions.create(
#             model=model,
#             messages=messages,
#             temperature=0
#         )
#         return {
#             "answer": response.choices[0].message.content,
#             "sources": input_documents
#         }
#     except Exception as e:
#         st.error(f"Could not find llm model: {str(e)}")
#         return None