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

WEAVIATE_CLASS_NAME = "DocumentConversationAlUsers"

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
        try:
            response = st.session_state.conversation({'question': user_input})
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}", icon="⚠️")
            return
        st.session_state.chat_history = response['chat_history']

    # Reverse the chat history to display the newest messages first
    for i, message in enumerate(reversed(st.session_state.chat_history)):
        if i % 2 != 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
    st.session_state.user_input = ""


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


def main():
    st.set_page_config(page_title="Converse with documents", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    try:
        weaviate_client = weaviate.connect_to_local()
    except WeaviateConnectionError:
        st.error('Cannot connect to the database!', icon="🚨")
        return

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    # Create or update schema
    if not weaviate_client.collections.exists(WEAVIATE_CLASS_NAME):
        weaviate_client.collections.create(WEAVIATE_CLASS_NAME,
                                           properties=[
                                               Property(name="title", data_type=DataType.TEXT),
                                               Property(name="body", data_type=DataType.TEXT),
                                               Property(name="fileName", data_type=DataType.TEXT)
                                           ])

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = WeaviateVectorStore(client=weaviate_client, index_name=WEAVIATE_CLASS_NAME, text_key="text",
                                      embedding=embeddings)

    if st.session_state.conversation is None:
        st.session_state.conversation = get_conversation_chain(vectorstore)

    st.header("Informatica :: Converse with documents :books:")

    question = st.text_input("Message:", value=st.session_state.user_input)
    if question:
        st.session_state.user_input = question
        handle_userinput(question)

    with st.sidebar:
        st.subheader("PDFs")
        pdf_files = st.file_uploader("Upload", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                store_pdf_content(pdf_extract_text(pdf_files), vectorstore)
                st.session_state.conversation = get_conversation_chain(vectorstore)

        st.write("Loaded documents")
        for file_name in get_all_files(weaviate_client, WEAVIATE_CLASS_NAME):
            col1, col2 = st.columns([4, 1])
            col1.write(file_name)
            if col2.button("X", key=f"remove_{file_name}"):
                remove_file_and_embeddings(file_name, weaviate_client, WEAVIATE_CLASS_NAME)


if __name__ == '__main__':
    main()
