import os
import weaviate
import traceback
import streamlit as st
from pypdf import PdfReader

# Optional: better PDF text extraction (handles many manuals better than pypdf)
try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
# (removed ConversationalRetrievalChain; using single-call RAG)
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from weaviate.exceptions import WeaviateConnectionError
from weaviate.classes.config import Property, DataType
from weaviate.classes.query import Filter

from html_templates import css, bot_template, user_template
from streamhandler import StreamHandler

WEAVIATE_CLASS_NAME = "DocumentConversationAlUsers"
# Default to a small model that fits better in 4GB VRAM; allow overriding via env.
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b-instruct")
EMBEDDER_MODEL = "nomic-embed-text"
# In k8s, use the service DNS name; locally we need to port-forward and set OLLAMA_URL=http://localhost:11434
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama.default.svc.cluster.local:11434")
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "weaviate.default.svc.cluster.local")


def pdf_extract_text(pdf_files: list) -> dict:
    """Extract text from uploaded PDFs.

    We try PyMuPDF first (usually better for manuals / weird encodings).
    Fallback to pypdf.
    """
    pdf_texts: dict[str, str] = {}

    for pdf in pdf_files:
        text = ""

        # Streamlit upload objects behave like file-like objects.
        if fitz is not None:
            try:
                pdf.seek(0)
                data = pdf.read()
                doc = fitz.open(stream=data, filetype="pdf")
                for page in doc:
                    text += page.get_text("text") or ""
                doc.close()
            except Exception:
                # Fall back to pypdf
                text = ""

        if not text:
            try:
                pdf.seek(0)
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
            except Exception:
                text = ""

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


def get_llm():
    """Chat LLM used for answering.

    We intentionally do a single LLM call per user message (no condense/rephrase step),
    because the extra chain step was showing up as an "intermediary question".
    """
    keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

    # Match what Ollama is willing to use; qwen2.5:3b-instruct typically supports 4096.
    num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "4096"))

    return ChatOllama(
        model=LLM_MODEL,
        temperature=0,
        base_url=OLLAMA_URL,
        streaming=True,
        keep_alive=keep_alive,
        num_ctx=num_ctx,
    )


def build_rag_prompt(question: str, docs, chat_messages) -> str:
    # Keep history short to reduce prompt size / latency
    history_lines = []
    for m in chat_messages[-6:]:
        t = getattr(m, "type", "")
        if t == "human":
            history_lines.append(f"User: {m.content}")
        elif t == "ai":
            history_lines.append(f"Assistant: {m.content}")

    context_blocks = []
    for i, d in enumerate(docs):
        src = ""
        try:
            src = d.metadata.get("fileName", "")
        except Exception:
            src = ""
        header = f"[Source {i+1}{': ' + src if src else ''}]"
        context_blocks.append(header + "\n" + (d.page_content or ""))

    history_text = "\n".join(history_lines).strip()
    context_text = "\n\n".join(context_blocks).strip() or "(no relevant context retrieved)"

    return (
        "You are a helpful assistant answering questions about the user's documents.\n"
        "Use ONLY the provided context when possible; if the context is insufficient, say so and ask a clarifying question.\n\n"
        + ("CHAT HISTORY:\n" + history_text + "\n\n" if history_text else "")
        + "CONTEXT:\n" + context_text + "\n\n"
        + "QUESTION:\n" + question + "\n\n"
        + "ANSWER:\n"
    )


def handle_userinput(user_input: str, chat_container) -> None:
    # Safeguard against Streamlit reruns causing duplicate sends.
    # If the same question is already in-flight or was just processed, do nothing.
    if st.session_state.get("_inflight_question") == user_input:
        return
    if st.session_state.get("_last_processed_question") == user_input:
        return

    if not st.session_state.vectorstore or not st.session_state.llm:
        with chat_container:
            st.write(bot_template.replace(
                "{{MSG}}", "Please process some documents first!"), unsafe_allow_html=True)
        st.session_state["_last_processed_question"] = user_input
        return

    st.session_state["_inflight_question"] = user_input

    # Persist the user's message first
    st.session_state.chat_messages.append(HumanMessage(content=user_input))

    with chat_container:
        st.write(user_template.replace("{{MSG}}", user_input), unsafe_allow_html=True)

        # Create a placeholder for the AI response (we'll show a typing animation until tokens arrive)
        response_placeholder = st.empty()
        response_placeholder.markdown(
            bot_template.replace(
                "{{MSG}}",
                '<span class="typing"><span class="dot"></span><span class="dot"></span><span class="dot"></span></span>'
            ),
            unsafe_allow_html=True
        )

    try:
        stream_handler = StreamHandler(response_placeholder)

        # Pass chat history explicitly
        history = st.session_state.get("chat_messages", [])

        docs = st.session_state.vectorstore.similarity_search(user_input, k=4)
        prompt = build_rag_prompt(user_input, docs, history)

        llm = st.session_state.llm
        response = llm.invoke(
            prompt,
            config={"callbacks": [stream_handler]},
        )
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}", icon="⚠️")
        st.error(traceback.format_exc())
        st.session_state["_inflight_question"] = None
        return

    answer_text = (stream_handler.text or "").strip()
    if not answer_text:
        # Fallback if for some reason streaming didn't populate
        answer_text = getattr(response, "content", "") if response is not None else ""

    st.session_state.chat_messages.append(AIMessage(content=answer_text))

    st.session_state["_last_processed_question"] = user_input
    st.session_state["_inflight_question"] = None


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
            # Use service DNS (FQDN) by default; allow overriding via env.
            weaviate_host = os.getenv('WEAVIATE_URL', WEAVIATE_URL)
            weaviate_host = weaviate_host.replace('http://', '').replace('https://', '').strip('/')

            http_port = int(os.getenv('WEAVIATE_HTTP_PORT', '8080'))
            grpc_port = int(os.getenv('WEAVIATE_GRPC_PORT', '50051'))
            http_secure = os.getenv('WEAVIATE_HTTP_SECURE', 'false').lower() == 'true'
            grpc_secure = os.getenv('WEAVIATE_GRPC_SECURE', 'false').lower() == 'true'

            st.session_state.weaviate_client = weaviate.connect_to_custom(
                http_host=weaviate_host,
                http_port=http_port,
                http_secure=http_secure,
                grpc_host=weaviate_host,
                grpc_port=grpc_port,
                grpc_secure=grpc_secure,
            )
        except WeaviateConnectionError:
            st.error('Cannot connect to the database!', icon="🚨")
            return None
    return st.session_state.weaviate_client


def main():
    st.set_page_config(page_title="Informatica | Converse with documents",
                       page_icon=":books:", initial_sidebar_state="collapsed")
    st.write(css, unsafe_allow_html=True)

    # (conversation chain removed; we use single-call RAG)
    if "llm" not in st.session_state:
        st.session_state.llm = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    # Chat state (we store it ourselves; avoids LangChain memory deprecations)
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Back-compat (old key no longer used)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Rerun/submit safeguards
    if "_inflight_question" not in st.session_state:
        st.session_state._inflight_question = None
    if "_last_processed_question" not in st.session_state:
        st.session_state._last_processed_question = None

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

    embeddings = OllamaEmbeddings(
        base_url=OLLAMA_URL,
        model=EMBEDDER_MODEL,
    )
    vectorstore = WeaviateVectorStore(client=weaviate_client, index_name=WEAVIATE_CLASS_NAME, text_key="text",
                                      embedding=embeddings)

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = vectorstore
    else:
        st.session_state.vectorstore = vectorstore

    if "llm" not in st.session_state or st.session_state.llm is None:
        st.session_state.llm = get_llm()

    left_co, cent_co, last_co = st.columns(3)
    with cent_co:
        st.image("static/logo.png")

    st.logo("static/logo.png")
    st.header("Informatica :: Converse with documents :books:")

    chat_container = st.container()

    with chat_container:
        # Render chat history (chronological) from Streamlit state.
        for message in st.session_state.get("chat_messages", []):
            msg_type = getattr(message, "type", "")
            if msg_type == "human":
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            elif msg_type == "ai":
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

    # Chat input: use a form so we only submit on Enter / Send (not on blur / rerun)
    with st.form(key="chat_form", clear_on_submit=True):
        question = st.text_input("Message:", key="user_input")
        submitted = st.form_submit_button("Send")

    if submitted and question and question.strip():
        handle_userinput(question.strip(), chat_container)

    with st.sidebar:
        st.subheader("PDFs")
        pdf_files = st.file_uploader("Upload", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                store_pdf_content(pdf_extract_text(pdf_files), vectorstore)
                st.session_state.vectorstore = vectorstore
                st.session_state.llm = get_llm()

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
