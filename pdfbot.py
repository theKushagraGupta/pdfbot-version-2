from dotenv import load_dotenv                                                                                          # To load the environment variables

import streamlit as st                                                                                                  # Make a test website

from PyPDF2 import PdfReader                                                                                            # For reading the PDF


from langchain.text_splitter import RecursiveCharacterTextSplitter                                                      # To make chunks
import os                                                                                                               # To interact with the Operating System
from langchain_google_genai import GoogleGenerativeAIEmbeddings                                                         # For converting PDF text to vecotrs
from langchain.vectorstores import FAISS                                                                                # For Vector Embeddings (Facebook AI)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain                                                           # For question and answers
from langchain.prompts import PromptTemplate                                                                            # For prompts

import google.generativeai as genai                                             

load_dotenv()                                                                                                           # Loading the environment variables

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))                                                                  # Configuring Gemini API Key

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_chunks(text):                                                                                                   # To split the whole pdf content into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 250)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):                                                                                      # To convert the chunks into vectors for Semantic Search
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")                                           # GoogleAI Provides the method for embedding
    vector_store = FAISS.from_texts(text_chunks, embedding = embeddings)                                                # FAISS Does the embedding
    vector_store.save_local("FAISS_embeddings")                                                                         # Save the vector file with the name "FAISS_embeddings"

def get_chain():                                                                                                        # To get the conversational chain and the Prompt Template tells the model how to behave                             
    prompt_template = """
    Answert the question as accurately as possible from the provided context, make sure to provide all the necessary details, and never provide a wrong answer\n\n
    Context:\n {context}?\n
    Question:\n {question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model = "gemini-pro", temperature = 0.3)                                              # Bring the gemini-pro LLM model in the play

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])                       # Assigning the prompt and input variables

    chain = load_qa_chain(model, chain_type = "stuff", prompt = prompt)                                                  # Initiate the chain
    return chain

def user_input(user_query):                                                                                              # To take the user's question
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")                                           # Convert the user's query into embeddings for semantic search

    pdf_vector_store = FAISS.load_local("FAISS_embeddings", embeddings, allow_dangerous_deserialization = True)                                                  # Bring the vector store of the pdf uploaded by the user
    semantic_search = pdf_vector_store.similarity_search(user_query)                                                     # Perform similarity search

    chain = get_chain()                                                                                                  # Bring the chain to get the response from the LLM model
    response = chain(
        {"input_documents": semantic_search, "question": user_query},
        return_only_outputs= True
    )

    ai_response = response["output_text"]                                                                     # Print the response using streamlit
    st.write(ai_response)

    with st.sidebar:
        st.title("Chat History:")
        # Initialize session state for chat history if it doesn't exist
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []
        st.session_state['chat_history'].append(("You", user_query))
        st.session_state['chat_history'].append(("Bot", ai_response))
    
        for role, text in st.session_state['chat_history']:
            st.write(f"{role}: {text}")

def main():
    st.set_page_config("Metis")
    st.header("PDF ChatBot 📑")

    user_question = st.text_input("Ask something from the PDF files you have provided")
        

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("PDF Upload Section:")
        pdf_docs = st.file_uploader("Upload the PDF files here and click Submit", accept_multiple_files = True)
        if st.button("Submit"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()