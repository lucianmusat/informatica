import traceback

import streamlit as st
import weaviate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
from weaviate.classes.config import DataType, Property
from weaviate.exceptions import WeaviateConnectionError

import prompts
import rag
from config import (
    EMBEDDER_MODEL,
    LLM_MODEL,
    MAX_HISTORY_MESSAGES,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_NUM_CTX,
    OLLAMA_TEMPERATURE,
    OLLAMA_URL,
    RETRIEVAL_K,
    RETRIEVAL_MAX_DISTANCE,
    WEAVIATE_CLASS_NAME,
    WEAVIATE_GRPC_PORT,
    WEAVIATE_GRPC_SECURE,
    WEAVIATE_HTTP_PORT,
    WEAVIATE_HTTP_SECURE,
    WEAVIATE_URL,
)
from html_templates import TYPING_INDICATOR, css
from pdf_utils import extract_documents

LOGO = "static/logo.png"
BOT_AVATAR = "static/bot.png"
USER_AVATAR = "static/user.png"


def turn(role: str, index: int):
    """A chat message wrapped in a keyed container.

    Streamlit renders *custom image* avatars as stChatMessageAvatarCustom for both
    roles, so the avatar testid can't distinguish them in CSS. st.container(key=...)
    emits an "st-key-<key>" class on the wrapper, which the stylesheet keys off to
    tint the user's turn only.
    """
    return st.container(key=f"{role}turn-{index}")


# --------------------------------------------------------------------------
# Resources
#
# st.cache_resource keeps one client per server process. The previous version
# stashed the Weaviate client in session_state and closed it at the end of every
# run, so each rerun reconnected a client it had just torn down.
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_weaviate_client():
    host = WEAVIATE_URL.replace("http://", "").replace("https://", "").strip("/")
    client = weaviate.connect_to_custom(
        http_host=host,
        http_port=WEAVIATE_HTTP_PORT,
        http_secure=WEAVIATE_HTTP_SECURE,
        grpc_host=host,
        grpc_port=WEAVIATE_GRPC_PORT,
        grpc_secure=WEAVIATE_GRPC_SECURE,
    )
    if not client.collections.exists(WEAVIATE_CLASS_NAME):
        client.collections.create(
            WEAVIATE_CLASS_NAME,
            properties=[
                # "text" is the property WeaviateVectorStore writes chunks into.
                Property(name="text", data_type=DataType.TEXT),
                Property(name="fileName", data_type=DataType.TEXT),
            ],
        )
    return client


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return OllamaEmbeddings(base_url=OLLAMA_URL, model=EMBEDDER_MODEL)


@st.cache_resource(show_spinner=False)
def get_vectorstore():
    return WeaviateVectorStore(
        client=get_weaviate_client(),
        index_name=WEAVIATE_CLASS_NAME,
        text_key="text",
        embedding=get_embeddings(),
    )


@st.cache_resource(show_spinner=False)
def get_llm(model: str, num_ctx: int, temperature: float):
    return ChatOllama(
        model=model,
        base_url=OLLAMA_URL,
        temperature=temperature,
        num_ctx=num_ctx,
        keep_alive=OLLAMA_KEEP_ALIVE,
        streaming=True,
    )


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------

def gather_context(question: str, mode: str) -> list[rag.Hit]:
    """Retrieved excerpts for this turn, honouring the sidebar retrieval mode.

    "Auto" keeps only chunks within RETRIEVAL_MAX_DISTANCE of the question, so a
    message that has nothing to do with the library contributes no context at all.
    """
    if mode == "Never":
        return []

    try:
        hits = rag.retrieve(
            get_weaviate_client(),
            get_embeddings(),
            WEAVIATE_CLASS_NAME,
            question,
            RETRIEVAL_K,
        )
    except Exception:
        # A retrieval failure should degrade to plain chat, not break the reply.
        return []

    if mode == "Always":
        return hits
    return [h for h in hits if h.distance <= RETRIEVAL_MAX_DISTANCE]


def render_sources(hits: list[rag.Hit]) -> None:
    if not hits:
        return
    names = ", ".join(dict.fromkeys(h.file_name for h in hits))
    with st.expander(f"Sources · {names}"):
        for hit in hits:
            st.caption(f"**{hit.file_name}** — {hit.similarity:.0%} match")
            st.text(hit.text[:700] + ("…" if len(hit.text) > 700 else ""))


def stream_reply(messages) -> str:
    """Stream the model's answer into the page, returning the finished text."""
    placeholder = st.empty()
    placeholder.markdown(TYPING_INDICATOR, unsafe_allow_html=True)

    parts: list[str] = []
    for chunk in get_llm(LLM_MODEL, OLLAMA_NUM_CTX, OLLAMA_TEMPERATURE).stream(messages):
        text = chunk.content
        if not text:
            continue
        parts.append(text)
        placeholder.markdown("".join(parts) + " ▌")

    answer = "".join(parts).strip()
    placeholder.markdown(answer or "_(empty response)_")
    return answer


def answer(question: str, mode: str) -> None:
    index = len(st.session_state.messages)
    st.session_state.messages.append(HumanMessage(content=question))

    with turn("user", index), st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(question)

    with turn("assistant", index + 1), st.chat_message("assistant", avatar=BOT_AVATAR):
        try:
            hits = gather_context(question, mode)
            payload = [SystemMessage(content=prompts.with_context(hits))]
            payload += st.session_state.messages[-MAX_HISTORY_MESSAGES:]

            reply = stream_reply(payload)
            render_sources(hits)
        except Exception as exc:
            st.error(f"Something went wrong: {exc}", icon="⚠️")
            st.caption(traceback.format_exc())
            st.session_state.messages.pop()
            return

    st.session_state.messages.append(
        AIMessage(
            content=reply,
            additional_kwargs={
                "sources": [(h.file_name, h.text, h.distance) for h in hits]
            },
        )
    )


def replay_history() -> None:
    for index, message in enumerate(st.session_state.messages):
        if isinstance(message, HumanMessage):
            with turn("user", index), st.chat_message("user", avatar=USER_AVATAR):
                st.markdown(message.content)
        else:
            with turn("assistant", index), st.chat_message("assistant", avatar=BOT_AVATAR):
                st.markdown(message.content)
                stored = message.additional_kwargs.get("sources") or []
                render_sources([rag.Hit(text=t, file_name=f, distance=d) for f, t, d in stored])


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

def render_sidebar(client) -> str:
    with st.sidebar:
        st.subheader("Documents")

        uploads = st.file_uploader(
            "Upload PDFs", type=["pdf"], accept_multiple_files=True
        )
        if st.button("Process", use_container_width=True, disabled=not uploads):
            vectorstore = get_vectorstore()
            for name, text in extract_documents(uploads).items():
                if not text.strip():
                    st.warning(f"No extractable text in {name} — is it a scan?")
                    continue
                with st.spinner(f"Indexing {name}…"):
                    try:
                        chunks = rag.store_document(vectorstore, name, text)
                        st.success(f"{name} — {chunks} chunks")
                    except Exception as exc:
                        st.error(f"{name}: {exc}")

        documents = []
        try:
            documents = rag.list_documents(client, WEAVIATE_CLASS_NAME)
        except Exception as exc:
            st.error(f"Could not list documents: {exc}")

        if documents:
            st.caption("In your library")
            for name in documents:
                row, remove = st.columns([5, 1])
                row.write(name)
                if remove.button("✕", key=f"rm_{name}", help=f"Remove {name}"):
                    with st.spinner(f"Removing {name}…"):
                        deleted = rag.delete_document(client, WEAVIATE_CLASS_NAME, name)
                    st.toast(f"Removed {name} ({deleted} chunks)")
                    st.rerun()
        else:
            st.caption("No documents yet — the assistant still answers normally.")

        st.divider()

        mode = st.radio(
            "Use documents",
            options=["Auto", "Always", "Never"],
            index=0,
            horizontal=True,
            help=(
                "Auto consults your library only when a passage is a close match "
                "for your message. Always forces retrieval every turn; Never turns "
                "it off entirely."
            ),
            disabled=not documents,
        )
        if not documents:
            mode = "Never"

        st.divider()

        if st.button("New chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.caption(f"Model · `{LLM_MODEL}`")
        # Vector search fails *silently* on a Weaviate server older than the
        # client expects (near_vector returns nothing at all, no error), so make
        # the server version visible rather than debuggable only from logs.
        try:
            st.caption(f"Weaviate · `{client.get_meta().get('version', '?')}`")
        except Exception:
            st.caption("Weaviate · `unreachable`")

    return mode


# --------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Informatica",
        page_icon=LOGO,
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    st.markdown(css, unsafe_allow_html=True)
    st.logo(LOGO)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    try:
        client = get_weaviate_client()
    except WeaviateConnectionError:
        st.error("Cannot reach the document database.", icon="🚨")
        st.caption(
            "The assistant needs Weaviate for document storage. "
            f"Tried `{WEAVIATE_URL}:{WEAVIATE_HTTP_PORT}`."
        )
        return

    mode = render_sidebar(client)

    # st.chat_input pins itself to the bottom wherever it is called, so reading
    # it first lets this run know a question is incoming and skip the greeting.
    question = st.chat_input("Message Informatica…")

    if not st.session_state.messages and not question:
        # Keyed so the stylesheet can centre the whole block; see html_templates.
        with st.container(key="welcome"):
            st.image(LOGO, width=76)
            st.markdown("#### What can I help you with?")
            st.caption(
                "Ask me anything. Upload PDFs in the sidebar and I'll draw on them "
                "when they're relevant."
            )

    replay_history()

    if question and question.strip():
        answer(question.strip(), mode)


if __name__ == "__main__":
    main()
