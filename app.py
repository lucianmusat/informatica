import os
import weaviate
import traceback
import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from weaviate.exceptions import WeaviateConnectionError
from weaviate.classes.config import Property, DataType
from weaviate.classes.query import Filter

from html_templates import css, bot_template, user_template
from streamhandler import StreamHandler

WEAVIATE_CLASS_NAME = "DocumentConversationAlUsers"
LLM_MODEL = "mistral"
EMBEDDER_MODEL = "nomic-embed-text"
OLLAMA_URL = "http://ollama.default.svc.cluster.local:11434"
# OLLAMA_URL = "http://localhost:11434"
WEAVIATE_URL = "weaviate"
# WEAVIATE_URL = "localhost"


def pdf_extract_text(pdf_files: list) -> dict:
    pdf_texts = {}
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        pdf_texts[pdf.name] = text  # Associate text with file name
    return pdf_texts


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
        model=LLM_MODEL,
        temperature=0,
        base_url=OLLAMA_URL,
        streaming=True
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

    st.write(user_template.replace("{{MSG}}", user_input), unsafe_allow_html=True)

    # Create a placeholder for the AI response
    response_placeholder = st.empty()

    with st.spinner('Processing your question...'):
        try:
            stream_handler = StreamHandler(response_placeholder)

            # Use the callbacks parameter in the __call__ method
            response = st.session_state.conversation.__call__(
                {'question': user_input},
                callbacks=[stream_handler]
            )
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}", icon="⚠️")
            st.error(traceback.format_exc())
            return

        st.session_state.chat_history = response['chat_history']


def remove_file_and_embeddings(file_name: str, client, class_name: str):
    with st.spinner(f"Removing {file_name}..."):
        try:
            collection = client.collections.get(class_name)
            file_filter = Filter.by_property("fileName").equal(file_name)
            result = collection.data.delete_many(where=file_filter)
            if file_name in st.session_state.uploaded_files:
                st.session_state.uploaded_files.remove(file_name)
            st.success(
                f"Successfully removed {file_name} and its related embeddings. {result.matches} object(s) deleted.")
        except Exception as e:
            st.error(f"Error removing {file_name}: {str(e)}")
            st.error(traceback.format_exc())


def store_pdf_content(pdf_texts, vectorstore):
    for file_name, text in pdf_texts.items():
        try:
            text_chunks = get_text_chunks(text)
            metadata = [{"fileName": file_name} for _ in text_chunks]  # Include file name as metadata
            vectorstore.add_texts(text_chunks, metadatas=metadata)  # Add texts with metadata
            if file_name not in st.session_state.uploaded_files:
                st.session_state.uploaded_files.append(file_name)
            st.success(f"Successfully processed and stored {file_name}")
        except Exception as e:
            st.error(f"Error processing {file_name}: {str(e)}")
            st.error(traceback.format_exc())


def get_all_files(client, class_name: str) -> list[str]:
    try:
        collection = client.collections.get(class_name)
        file_names = set()
        for item in collection.iterator():
            if "fileName" in item.properties:
                file_name = item.properties["fileName"]
                if file_name is not None:
                    file_names.add(file_name)
        return list(file_names)
    except Exception as e:
        st.error(f"Error fetching files: {e}")
        return []


def get_weaviate_client():
    if 'weaviate_client' not in st.session_state:
        try:
            weaviate_url = os.getenv('WEAVIATE_URL', WEAVIATE_URL)
            st.session_state.weaviate_client = weaviate.connect_to_custom(
                http_host=weaviate_url,
                http_port=8080,
                http_secure=False,
                grpc_host=weaviate_url,
                grpc_port=50051,
                grpc_secure=False
            )
        except WeaviateConnectionError:
            st.error('Cannot connect to the database!', icon="🚨")
            return None
    return st.session_state.weaviate_client


def main():
    st.set_page_config(page_title="Informatica | Converse with documents",
                       page_icon=":books:", initial_sidebar_state="collapsed")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    weaviate_client = get_weaviate_client()
    if not weaviate_client:
        return
    weaviate_client.connect()

    # Create or update schema
    if not weaviate_client.collections.exists(WEAVIATE_CLASS_NAME):
        weaviate_client.collections.create(WEAVIATE_CLASS_NAME,
                                           properties=[
                                               Property(name="title", data_type=DataType.TEXT),
                                               Property(name="body", data_type=DataType.TEXT),
                                               Property(name="fileName", data_type=DataType.TEXT)
                                           ])

    embeddings = OllamaEmbeddings(base_url=OLLAMA_URL, model=EMBEDDER_MODEL)
    vectorstore = WeaviateVectorStore(client=weaviate_client, index_name=WEAVIATE_CLASS_NAME, text_key="text",
                                      embedding=embeddings)

    if st.session_state.conversation is None:
        st.session_state.conversation = get_conversation_chain(vectorstore)

    left_co, cent_co, last_co = st.columns(3)
    with cent_co:
        st.image("static/logo.png")

    st.logo("static/logo.png")
    st.header("Informatica :: Converse with documents :books:")

    question = st.text_input("Message:", key="user_input")
    if question:
        handle_userinput(question)

    if st.session_state.chat_history:
        # Exclude the last message pair, because it has been already displayed by the callback
        for i, message in enumerate(reversed(st.session_state.chat_history[:-2])):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("PDFs")
        pdf_files = st.file_uploader("Upload", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                store_pdf_content(pdf_extract_text(pdf_files), vectorstore)
                st.session_state.conversation = get_conversation_chain(vectorstore)

        st.write("Available documents")
        for file_name in get_all_files(weaviate_client, WEAVIATE_CLASS_NAME):
            col1, col2 = st.columns([4, 1])
            col1.write(file_name)
            if col2.button("X", key=f"remove_{file_name}"):
                remove_file_and_embeddings(file_name, weaviate_client, WEAVIATE_CLASS_NAME)

    if st.session_state.get('weaviate_client'):
        st.session_state.weaviate_client.close()

if __name__ == '__main__':
    main()
