import weaviate
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from html_templates import css, bot_template, user_template


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
    llm = ChatOllama(
        model="mistral",
        temperature=0,
    )
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_input: str) -> None:
    if not st.session_state.conversation:
        st.write(bot_template.replace(
            "{{MSG}}", "Please process some documents first!"), unsafe_allow_html=True)
        return
    with st.spinner('Processing your question...'):
        response = st.session_state.conversation({'question': user_input})
        st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Converse with documents", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    weaviate_client = weaviate.connect_to_local()

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "files_processed" not in st.session_state:
        st.session_state.files_processed = False
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    # TODO: reconnect to weaviate and hook it back to previous embeddings
    if st.session_state.conversation is None:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = WeaviateVectorStore(client=weaviate_client, index_name=None, text_key="anonymous", embedding=embeddings)
        st.session_state.conversation = get_conversation_chain(vectorstore)

    st.header("Informatica :: Converse with documents :books:")

    question = st.text_input("Message:", value=st.session_state.user_input)
                                         # disabled=not st.session_state.files_processed)
    if question:
        st.session_state.user_input = question
        handle_userinput(question)

    with st.sidebar:
        st.subheader("PDFs")
        pdf_files = st.file_uploader("Upload", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                text_chunks = get_text_chunks(pdf_extract_text(pdf_files))
                embeddings = OllamaEmbeddings(model="nomic-embed-text",)
                vectorstore = WeaviateVectorStore.from_texts(text_chunks, embeddings, client=weaviate_client)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.session_state.files_processed = True


if __name__ == '__main__':
    main()
