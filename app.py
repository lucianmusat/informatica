import json
import logging
import threading
import time
import traceback
import urllib.request
from dataclasses import dataclass

import streamlit as st
import weaviate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from cassandra.cluster import Cluster
from cassandra.policies import DCAwareRoundRobinPolicy
from langchain_weaviate.vectorstores import WeaviateVectorStore
from weaviate.exceptions import WeaviateConnectionError

import backends
import prompts
import rag
from access import Access, check_access
from notify import alert_new_user
from users import APPROVED, BLOCKED, PENDING, CassandraUserStore
from config import (
    ADMIN_EMAIL,
    CASSANDRA_HOSTS,
    CASSANDRA_KEYSPACE,
    CASSANDRA_PORT,
    EMBEDDER_MODEL,
    LLM_MODEL,
    LLM_MODEL_BIG,
    LLM_REASONING,
    MAX_HISTORY_MESSAGES,
    OLLAMA_BIG_COOLDOWN,
    OLLAMA_BIG_NAME,
    OLLAMA_BIG_PROBE_TTL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_CTX_BIG,
    OLLAMA_SMALL_NAME,
    OLLAMA_TEMPERATURE,
    OLLAMA_URL,
    OLLAMA_URL_BIG,
    RETRIEVAL_K,
    RETRIEVAL_MAX_DISTANCE,
    WEAVIATE_CLASS_NAME,
    WEAVIATE_GRPC_PORT,
    WEAVIATE_GRPC_SECURE,
    WEAVIATE_HTTP_PORT,
    WEAVIATE_HTTP_SECURE,
    WEAVIATE_URL,
)
from html_templates import ON_SEND_SCRIPT, THINKING_INDICATOR, TYPING_INDICATOR, css
from pdf_utils import extract_documents
from timing import format_duration

LOGO = "static/logo.png"
BOT_AVATAR = "static/bot.png"
USER_AVATAR = "static/user.png"

log = logging.getLogger("informatica")


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
        # "text" is the property WeaviateVectorStore writes chunks into.
        client.collections.create(WEAVIATE_CLASS_NAME, properties=rag.collection_properties())
    else:
        # Collections from before documents were per-user: add the owner field
        # and hand the existing documents to the admin.
        rag.ensure_owner_property(client, WEAVIATE_CLASS_NAME)
        moved = rag.assign_unowned(client, WEAVIATE_CLASS_NAME, ADMIN_EMAIL)
        if moved:
            log.warning("assigned %d unowned chunks to %s", moved, ADMIN_EMAIL)
    return client


@st.cache_resource(show_spinner=False)
def get_user_store() -> CassandraUserStore:
    """Raises if Cassandra is unreachable; cache_resource does not cache failures,
    so the next rerun simply tries again."""
    cluster = Cluster(
        CASSANDRA_HOSTS,
        port=CASSANDRA_PORT,
        connect_timeout=3,
        load_balancing_policy=DCAwareRoundRobinPolicy(local_dc="dc1"),
        protocol_version=5,
    )
    store = CassandraUserStore(cluster.connect(), CASSANDRA_KEYSPACE)
    store.ensure_schema()
    return store


def current_owner() -> str:
    """Email of the signed-in user; every document operation is scoped to it."""
    return st.session_state["user_email"]


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


# --------------------------------------------------------------------------
# Chat backend selection: big (optional, ephemeral) with fallback to small
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Backend:
    name: str
    url: str
    model: str
    num_ctx: int
    # True streams the model's reasoning separately (used only to show
    # "Thinking..."); None leaves it to the model and must be used for models
    # without thinking support, which reject reasoning=True.
    reasoning: bool | None = None


SMALL = Backend(OLLAMA_SMALL_NAME, OLLAMA_URL, LLM_MODEL, OLLAMA_NUM_CTX, reasoning=LLM_REASONING)
BIG = Backend(OLLAMA_BIG_NAME, OLLAMA_URL_BIG, LLM_MODEL_BIG, OLLAMA_NUM_CTX_BIG, reasoning=True)

# A cold load of the big model from disk takes over a minute; give up after this.
BIG_WARMUP_TIMEOUT = 300


@st.cache_resource(show_spinner=False)
def _big_state() -> dict:
    """Process-wide, shared across sessions."""
    return {"down_until": 0.0, "warming_until": 0.0}


def _get_json(path: str) -> dict:
    with urllib.request.urlopen(f"{OLLAMA_URL_BIG}{path}", timeout=1) as resp:
        return json.load(resp)


@st.cache_data(ttl=OLLAMA_BIG_PROBE_TTL, show_spinner=False)
def _probe_big() -> str:
    """"ready" (model in VRAM), "cold" (pulled but not loaded) or "down".

    Checking /api/tags avoids routing to a freshly started node that is still
    downloading the model; checking /api/ps avoids making a user wait for a cold
    load, which reads 17GB from disk.
    """
    try:
        if backends.model_loaded(_get_json("/api/ps"), LLM_MODEL_BIG, OLLAMA_NUM_CTX_BIG):
            return "ready"
        return "cold" if backends.model_pulled(_get_json("/api/tags"), LLM_MODEL_BIG) else "down"
    except Exception:
        return "down"


def _warm_big() -> None:
    """Load the big model in the background, once at a time across sessions.

    An empty /api/generate request loads the model without generating. It uses
    the same num_ctx as chat requests so the loaded copy is the one chat uses.
    """
    state = _big_state()
    if time.time() < state["warming_until"]:
        return
    state["warming_until"] = time.time() + BIG_WARMUP_TIMEOUT

    def run() -> None:
        body = json.dumps(backends.with_shift_disabled({
            "model": LLM_MODEL_BIG,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {"num_ctx": OLLAMA_NUM_CTX_BIG},
        })).encode()
        req = urllib.request.Request(
            f"{OLLAMA_URL_BIG}/api/generate", data=body,
            headers={"Content-Type": "application/json"},
        )
        started = time.time()
        try:
            with urllib.request.urlopen(req, timeout=BIG_WARMUP_TIMEOUT) as resp:
                resp.read()
            log.info("warmed %s on %s in %.0fs", LLM_MODEL_BIG, OLLAMA_BIG_NAME, time.time() - started)
        except Exception:
            log.warning("warm-up of %s failed", LLM_MODEL_BIG, exc_info=True)
        finally:
            state["warming_until"] = 0.0
            _probe_big.clear()

    threading.Thread(target=run, daemon=True, name="warm-big").start()


def big_status() -> str:
    """"off", "down", "warming" or "ready". A cold model starts warming."""
    if not OLLAMA_URL_BIG:
        return "off"
    if time.time() < _big_state()["down_until"]:
        return "down"
    status = _probe_big()
    if status == "cold":
        _warm_big()
        return "warming"
    return status


def mark_big_down() -> None:
    _big_state()["down_until"] = time.time() + OLLAMA_BIG_COOLDOWN
    _probe_big.clear()


def pick_backend() -> Backend:
    """Big only when its model is already loaded; never wait for a cold load."""
    return BIG if big_status() == "ready" else SMALL


@st.cache_resource(show_spinner=False)
def get_llm(url: str, model: str, num_ctx: int, temperature: float, reasoning: bool | None):
    return ChatOllama(
        model=model,
        base_url=url,
        temperature=temperature,
        num_ctx=num_ctx,
        keep_alive=OLLAMA_KEEP_ALIVE,
        reasoning=reasoning,
        # Same runner options as OpenClaw, so the shared model stays loaded.
        sync_client_kwargs={"transport": backends.NoShiftTransport()},
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
            owner=current_owner(),
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


def backend_label(backend: Backend) -> str:
    return f"{backend.model} · {backend.name}"

def _stream_from(backend: Backend, messages, placeholder) -> str:
    llm = get_llm(backend.url, backend.model, backend.num_ctx, OLLAMA_TEMPERATURE, backend.reasoning)
    parts: list[str] = []
    thinking = False
    for chunk in llm.stream(messages):
        text = chunk.content
        if not text:
            # Reasoning is never shown, only signalled, so a long think doesn't
            # look like a hang.
            if not parts and not thinking and chunk.additional_kwargs.get("reasoning_content"):
                placeholder.markdown(THINKING_INDICATOR, unsafe_allow_html=True)
                thinking = True
            continue
        parts.append(text)
        placeholder.markdown("".join(parts) + " ▌")
    return "".join(parts).strip()

def stream_reply(messages) -> tuple[str, Backend]:
    """Stream the model's answer into the page.

    Tries the big backend when it is up. If it fails at any point (the desktop
    was switched off or the GPU got claimed by a game mid-answer) the reply is
    restarted from scratch on the small backend, and the big one is skipped for
    OLLAMA_BIG_COOLDOWN seconds so later turns don't pay for the failure again.
    """
    placeholder = st.empty()
    placeholder.markdown(TYPING_INDICATOR, unsafe_allow_html=True)

    backend = pick_backend()
    try:
        answer = _stream_from(backend, messages, placeholder)
    except Exception:
        if backend is SMALL:
            raise
        mark_big_down()
        backend = SMALL
        placeholder.markdown(TYPING_INDICATOR, unsafe_allow_html=True)
        answer = _stream_from(backend, messages, placeholder)

    placeholder.markdown(answer or "_(empty response)_")
    return answer, backend


def answer(question: str, mode: str) -> None:
    started = time.monotonic()  # the caption shows the whole wait, retrieval included
    index = len(st.session_state.messages)
    st.session_state.messages.append(HumanMessage(content=question))

    with turn("user", index), st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(question)

    with turn("assistant", index + 1), st.chat_message("assistant", avatar=BOT_AVATAR):
        try:
            hits = gather_context(question, mode)
            payload = [SystemMessage(content=prompts.with_context(hits))]
            payload += st.session_state.messages[-MAX_HISTORY_MESSAGES:]

            reply, backend = stream_reply(payload)
            caption = f"{backend_label(backend)} · {format_duration(time.monotonic() - started)}"
            st.caption(caption)
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
                "sources": [(h.file_name, h.text, h.distance) for h in hits],
                "backend": caption,
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
                if label := message.additional_kwargs.get("backend"):
                    st.caption(label)
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
                        chunks = rag.store_document(vectorstore, name, text, owner=current_owner())
                        st.success(f"{name} — {chunks} chunks")
                    except Exception as exc:
                        st.error(f"{name}: {exc}")

        documents = []
        try:
            documents = rag.list_documents(client, WEAVIATE_CLASS_NAME, current_owner())
        except Exception as exc:
            st.error(f"Could not list documents: {exc}")

        if documents:
            st.caption("In your library")
            for name in documents:
                row, remove = st.columns([5, 1])
                row.write(name)
                if remove.button("✕", key=f"rm_{name}", help=f"Remove {name}"):
                    with st.spinner(f"Removing {name}…"):
                        deleted = rag.delete_document(client, WEAVIATE_CLASS_NAME, name, current_owner())
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

        current = pick_backend()
        note = f" ({OLLAMA_BIG_NAME} warming up)" if big_status() == "warming" else ""
        st.caption(f"Model · `{current.model}` on {current.name}{note}")
        # Vector search fails *silently* on a Weaviate server older than the
        # client expects (near_vector returns nothing at all, no error), so make
        # the server version visible rather than debuggable only from logs.
        try:
            st.caption(f"Weaviate · `{client.get_meta().get('version', '?')}`")
        except Exception:
            st.caption("Weaviate · `unreachable`")

    return mode


# --------------------------------------------------------------------------
# Sign-in and access
# --------------------------------------------------------------------------

def render_landing() -> None:
    with st.container(key="welcome"):
        st.image(LOGO, width=76)
        st.markdown("#### Informatica")
        st.caption("A private assistant. Sign in with Google to request access.")
        if st.button("Sign in with Google", type="primary"):
            try:
                st.login()
            except Exception as exc:
                log.exception("sign-in is not configured")
                st.error(f"Sign-in is not available right now ({type(exc).__name__}).")


WAITING_TEXT = {
    Access.PENDING: "Your access request has been sent. You'll be able to use "
                    "Informatica once it's approved.",
    # Deliberately the same as pending: no need to tell someone they are blocked.
    Access.BLOCKED: "Your access request has been sent. You'll be able to use "
                    "Informatica once it's approved.",
    Access.UNVERIFIED: "Your Google account's email address isn't verified, so "
                       "access can't be requested with it.",
    Access.UNAVAILABLE: "Access can't be checked right now. Please try again in a "
                        "few minutes.",
}


def render_waiting(access: Access) -> None:
    with st.container(key="welcome"):
        st.image(LOGO, width=76)
        st.markdown(f"#### Hi {st.user.get('given_name') or st.user.get('name') or ''}".rstrip())
        st.caption(WAITING_TEXT[access])
        st.caption(f"Signed in as {st.user.get('email')}")
        st.button("Log out", on_click=st.logout)


def gate() -> Access:
    """Render the sign-in or waiting screen unless this visitor may use the app.

    Runs before anything touches Ollama or Weaviate, so an unapproved visitor
    cannot spend GPU time or read documents.
    """
    if not st.user.is_logged_in:
        render_landing()
        return Access.UNVERIFIED

    email = (st.user.get("email") or "").strip().lower()
    if st.session_state.get("user_email") != email:
        # A different account in the same browser session: start clean.
        st.session_state.messages = []
        st.session_state.pop("registered_for", None)
    st.session_state.user_email = email

    first_check = st.session_state.get("registered_for") != email
    access = check_access(
        email,
        bool(st.user.get("email_verified", False)),
        get_user_store,
        ADMIN_EMAIL,
        register=first_check,
        name=st.user.get("name"),
        picture=st.user.get("picture"),
        on_new_user=alert_new_user,
    )
    if first_check and access is not Access.UNAVAILABLE:
        st.session_state.registered_for = email

    if not access.allowed:
        render_waiting(access)
    return access


def render_account() -> None:
    with st.sidebar:
        st.caption(f"Signed in as {st.session_state.user_email}")
        st.button("Log out", on_click=st.logout, use_container_width=True)


# --------------------------------------------------------------------------
# Admin: access requests
# --------------------------------------------------------------------------

def _when(ts) -> str:
    return ts.strftime("%Y-%m-%d %H:%M") if ts else "never"


def _decide(email: str, status: str) -> None:
    get_user_store().set_status(email, status, ADMIN_EMAIL)
    st.toast(f"{email}: {status}")


ADMIN_SECTIONS = [
    (PENDING, "Waiting for approval", [("Approve", APPROVED), ("Block", BLOCKED)]),
    (APPROVED, "Approved", [("Revoke", BLOCKED)]),
    (BLOCKED, "Blocked", [("Approve", APPROVED)]),
]


def admin_page() -> None:
    # Navigation only offers this page to the admin; check again anyway.
    if st.session_state.get("user_email") != ADMIN_EMAIL:
        st.error("Not allowed.")
        return

    st.markdown("#### Access requests")
    try:
        users = [u for u in get_user_store().list_all() if u.email != ADMIN_EMAIL]
    except Exception as exc:
        st.error(f"Cannot reach the user database: {exc}")
        return

    for status, heading, actions in ADMIN_SECTIONS:
        group = sorted(
            (u for u in users if u.status == status),
            key=lambda u: u.requested_at or u.last_login_at,
            reverse=True,
        )
        st.markdown(f"**{heading}** ({len(group)})")
        if not group:
            st.caption("Nobody here.")
            continue
        for user in group:
            with st.container(border=True):
                info, buttons = st.columns([3, 1])
                info.markdown(f"**{user.name or user.email}**  \n{user.email}")
                detail = f"Requested {_when(user.requested_at)} · last login {_when(user.last_login_at)}"
                if user.decided_at:
                    detail += f" · {status} {_when(user.decided_at)}"
                info.caption(detail)
                for label, new_status in actions:
                    buttons.button(
                        label,
                        key=f"{new_status}-{user.email}",
                        on_click=_decide,
                        args=(user.email, new_status),
                        use_container_width=True,
                        type="primary" if new_status == APPROVED else "secondary",
                    )


def pending_count() -> int:
    try:
        return sum(u.status == PENDING for u in get_user_store().list_all())
    except Exception:
        return 0


# --------------------------------------------------------------------------
# Chat page and entry point
# --------------------------------------------------------------------------

def chat_page() -> None:
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
    st.html(ON_SEND_SCRIPT, unsafe_allow_javascript=True)

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


def main() -> None:
    st.set_page_config(
        page_title="Informatica",
        page_icon=LOGO,
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    st.markdown(css, unsafe_allow_html=True)
    st.logo(LOGO)

    access = gate()
    if not access.allowed:
        return

    pages = [st.Page(chat_page, title="Chat", icon=":material/chat:", default=True)]
    if access is Access.ADMIN:
        waiting = pending_count()
        pages.append(st.Page(
            admin_page,
            title=f"Access requests ({waiting})" if waiting else "Access requests",
            icon=":material/group:",
            url_path="access-requests",
        ))
    page = st.navigation(pages)
    render_account()
    page.run()


if __name__ == "__main__":
    main()
