from xml.dom.minidom import Document

import weaviate
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.vectorstores import VectorStore
from langchain_weaviate.vectorstores import WeaviateVectorStore


def pdf_extract_text(pdf_files: list) -> str:
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text) -> list[str]:
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_conversation_chain(vectorstore):
    llm = None
    # llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def get_vector_store(text_chunks: list[Document], embeddings) -> VectorStore:
    weaviate_client = weaviate.connect_to_local()
    return WeaviateVectorStore.from_documents(text_chunks, embeddings, client=weaviate_client)

def main():
    st.set_page_config(page_title="Converse with documents", page_icon=":books:")
    st.header("Informatica :: Converse with documents :books:")
    st.text_input("Message:")
    with st.sidebar:
        st.subheader("PDFs")
        pdf_files = st.file_uploader("Upload", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                text_chunks = get_text_chunks(pdf_extract_text(pdf_files))
                embeddings = None
                vectorstore = get_vector_store(text_chunks, embeddings)
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()
