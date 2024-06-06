import os
import streamlit as st
from openai import AzureOpenAI
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import pickle


# Ensure environment variables are set
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://kant-openai-copy.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "efcacfae66a34614b863784b8bfb3555"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"

client = AzureOpenAI(api_version="2023-03-15-preview")      #initialize the Azure OpenAi service
USER_NAME = "user"          
ASSISTANT_NAME = "assistant"  #username
model = "azure_openai_app"   #name of model in Auzer OpanAI service

st.title("TB Chat 007")  #Name of Application



def get_pdf_text(file):                        #get all text from pdf file
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    print("\n---------------------============TEXT DOCUMENT=========-----------------------\n")
    print(text)
    print("\n---------------------==================================-----------------------\n")
    input("continue")
    return text

def get_text_chunks(text):                   #divide text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
        length_function=len
    )
    
    chunks = text_splitter.split_text(text)
    print("\n---------------------============FIRST CHUNK=========-----------------------\n")
    print(chunks[0])
    print("\n---------------------============SECOND CHUNK=========----------------------\n")
    print(chunks[1])
    input("continue")
    return chunks

def create_embeddings(text_chunks):         #create vector embeddings for text chunks

    # for japanese text,use this model in paranthesis of hugging face---->model_name="intfloat/multilingual-e5-base"
    embeddings = HuggingFaceEmbeddings()#OpenAIEmbeddings()
    sample_text="Hello this is suleman"
    doc_result3 = embeddings.embed_documents([sample_text])
    print(f"Length: {len(doc_result3[0])}")
    print("\n------------------============FIRST CHUNK VECTOR=========--------------------\n")
    print(f"{doc_result3[0][:]}")

    input("continue")
    vector_base = FAISS.from_texts(text_chunks, embeddings)   
    return vector_base

def store_vector(vectors, file_name):      # store vectors as pickle file for now
    if os.path.exists(f"{file_name}.pkl"):
        with open(f"{file_name}.pkl", "rb") as f:
            vector_store = pickle.load(f)
    else:
        vector_store = vectors
        with open(f"{file_name}.pkl", "wb") as f:
            pickle.dump(vector_store, f)
    return vector_store




def response_chatgpt(user_msg: str, input_documents, chat_history: list = []):
    system_msg = """You are an Assistant. Answer the questions based only on the provided documents. If the information is not in the documents, say you don't know."""
    messages = [{"role": "system", "content": system_msg}]

    # If there is a chat log, add it to the messages list
    if len(chat_history) > 0:
        for chat in chat_history:
            messages.append({"role": chat["name"], "content": chat["msg"]})

    # Add user message to messages list
    messages.append({"role": USER_NAME, "content": user_msg})

    # Append input documents to the messages list
    for doc in input_documents:
        messages.append({"role": "user", "content": f"Document snippet:\n{doc}"})
    
    try:
        response = client.chat.completions.create(               #Azure Open Ai client called
            model=model,
            messages=messages,
            temperature=0
        )
        return response
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

# Sidebar for PDF upload
with st.sidebar:
    st.title('PDF Chat Loader')
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    if pdf:
        file_name = pdf.name[:-4]
        if os.path.exists(f"{file_name}.pkl"):
            st.write("PDF file already exists.")
        else:
            st.write("File stored successfully.")


# Initialize the session information that saves the chat log
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

    if pdf:                                                 # If pdf vector already exists, fetch from database
        file_name = pdf.name[:-4]
        if os.path.exists(f"{file_name}.pkl"):
            with open(f"{file_name}.pkl", "rb") as f:
                st.spinner("Fetching data from database")
                vectordb = pickle.load(f)
             
        else:                                              # If new pdf file then create vector db and store
            with st.spinner("Creating vector data"):
                text = get_pdf_text(pdf)
                chunks = get_text_chunks(text)
                vectordb = create_embeddings(chunks)
                vectordb = store_vector(vectordb, file_name)

        # Perform similarity search on the vector database
        docs = vectordb.similarity_search(query=user_msg, k=3)
        doc_texts = [doc.page_content for doc in docs]  # Correctly access the text content

        # Get the response from GPT
        response = response_chatgpt(user_msg, doc_texts, chat_history=st.session_state.chat_log)
        if response:
            with st.chat_message(ASSISTANT_NAME):
                assistant_msg = response.choices[0].message.content
                assistant_response_area = st.empty()
                assistant_response_area.write(assistant_msg)

            # Add chat logs to the session
            st.session_state.chat_log.append({"name": USER_NAME, "msg": user_msg})
            st.session_state.chat_log.append({"name": ASSISTANT_NAME, "msg": assistant_msg})

