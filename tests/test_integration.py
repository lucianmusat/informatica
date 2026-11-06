"""Against real Cassandra and Weaviate. Skipped unless INFORMATICA_IT=1.

    INFORMATICA_IT=1 CASSANDRA_HOSTS=<ip> WEAVIATE_URL=<ip> pytest tests/test_integration.py

Uses a throwaway keyspace and collection, both dropped afterwards.
"""

import hashlib
import os

import pytest

pytestmark = pytest.mark.skipif(os.getenv("INFORMATICA_IT") != "1", reason="integration")

KS = "informatica_it"
COLL = "InformaticaIt"


@pytest.fixture(scope="module")
def store():
    from cassandra.cluster import Cluster
    from cassandra.policies import DCAwareRoundRobinPolicy

    from config import CASSANDRA_HOSTS
    from users import CassandraUserStore

    cluster = Cluster(CASSANDRA_HOSTS, load_balancing_policy=DCAwareRoundRobinPolicy("dc1"),
                      protocol_version=5)
    session = cluster.connect()
    s = CassandraUserStore(session, KS)
    s.ensure_schema()
    yield s
    session.execute(f"DROP KEYSPACE IF EXISTS {KS}")
    cluster.shutdown()


def test_user_lifecycle(store):
    from users import APPROVED, BLOCKED, PENDING

    user, new = store.register_login("Bob@Example.com", "Bob", None)
    assert new and user.status == PENDING and user.email == "bob@example.com"
    user, new = store.register_login("bob@example.com", "Bobby", "pic")
    assert not new and user.name == "Bobby" and user.status == PENDING
    store.set_status("bob@example.com", APPROVED, "admin@x.com")
    assert store.get("BOB@example.com").status == APPROVED
    store.set_status("bob@example.com", BLOCKED, "admin@x.com")
    u = store.get("bob@example.com")
    assert u.status == BLOCKED and u.decided_by == "admin@x.com" and u.decided_at
    assert [x.email for x in store.list_all()] == ["bob@example.com"]


def test_chat_lifecycle(store):
    import time

    from chats import CassandraChatStore, Chat

    chats = CassandraChatStore(store.session, KS, keep=2, pinned_max=1)
    chats.ensure_schema()
    owner = "bob@example.com"

    def saved(question):
        chat = Chat(owner=owner, messages=[{"role": "user", "content": question},
                                           {"role": "assistant", "content": "ok"}])
        chats.save(chat)
        time.sleep(0.01)
        return chat

    first = saved("één")
    assert chats.set_pinned(owner, first.chat_id, True)
    others = [saved(f"q{i}") for i in range(3)]
    assert not chats.set_pinned(owner, others[0].chat_id, True)  # cap of 1
    assert [c.title for c in chats.list_for(owner)] == ["één", "q2", "q1"]

    assert chats.set_summary(owner, others[2].chat_id, "about q2", 2)
    chats.save(others[2])
    loaded = chats.load(owner, others[2].chat_id)
    assert loaded.messages[0]["content"] == "q2" and loaded.summary == "about q2"
    assert any(c.chat_id == others[2].chat_id and c.message_count == 2
               for c in chats.scan_for_summary())

    chats.delete(owner, others[2].chat_id)
    assert not chats.set_summary(owner, others[2].chat_id, "late", 2)
    assert chats.load(owner, others[2].chat_id) is None
    chats.delete_all(owner)
    assert chats.list_for(owner) == []


class FakeEmbeddings:
    """Deterministic vectors, so no Ollama is needed."""

    def _v(self, text):
        h = hashlib.sha256(text.encode()).digest()
        return [b / 255 for b in h[:16]]

    def embed_documents(self, texts):
        return [self._v(t) for t in texts]

    def embed_query(self, text):
        return self._v(text)


@pytest.fixture()
def weaviate_client():
    import weaviate

    from config import WEAVIATE_GRPC_PORT, WEAVIATE_HTTP_PORT, WEAVIATE_URL

    client = weaviate.connect_to_custom(
        http_host=WEAVIATE_URL, http_port=WEAVIATE_HTTP_PORT, http_secure=False,
        grpc_host=WEAVIATE_URL, grpc_port=WEAVIATE_GRPC_PORT, grpc_secure=False)
    client.collections.delete(COLL)
    yield client
    client.collections.delete(COLL)
    client.close()


def _store(client):
    from langchain_weaviate.vectorstores import WeaviateVectorStore

    return WeaviateVectorStore(client=client, index_name=COLL, text_key="text",
                               embedding=FakeEmbeddings())


def test_documents_are_private(weaviate_client):
    import rag

    weaviate_client.collections.create(COLL, properties=rag.collection_properties())
    vs, emb = _store(weaviate_client), FakeEmbeddings()
    rag.store_document(vs, "a.pdf", "alpha text", owner="bob@gmail.com")
    rag.store_document(vs, "a b.pdf", "beta text", owner="bob@gmail.com")
    rag.store_document(vs, "c.pdf", "gamma text", owner="x.bob@gmail.com")

    assert rag.list_documents(weaviate_client, COLL, "bob@gmail.com") == ["a b.pdf", "a.pdf"]
    assert rag.list_documents(weaviate_client, COLL, "x.bob@gmail.com") == ["c.pdf"]
    assert rag.list_documents(weaviate_client, COLL, "gmail.com") == []

    hits = rag.retrieve(weaviate_client, emb, COLL, "gamma text", 10, owner="bob@gmail.com")
    assert {h.file_name for h in hits} == {"a.pdf", "a b.pdf"}

    # Another user's file of the same name is untouched, and "a.pdf" does not
    # take "a b.pdf" with it.
    assert rag.delete_document(weaviate_client, COLL, "c.pdf", "bob@gmail.com") == 0
    assert rag.delete_document(weaviate_client, COLL, "a.pdf", "bob@gmail.com") == 1
    assert rag.list_documents(weaviate_client, COLL, "bob@gmail.com") == ["a b.pdf"]
    assert rag.list_documents(weaviate_client, COLL, "x.bob@gmail.com") == ["c.pdf"]


def test_existing_collection_is_migrated_to_admin(weaviate_client):
    from weaviate.classes.config import DataType, Property

    import rag

    # The pre-per-user schema.
    weaviate_client.collections.create(COLL, properties=[
        Property(name="text", data_type=DataType.TEXT),
        Property(name="fileName", data_type=DataType.TEXT),
    ])
    coll = weaviate_client.collections.get(COLL)
    coll.data.insert({"text": "old", "fileName": "old.pdf"}, vector=FakeEmbeddings().embed_query("old"))

    rag.ensure_owner_property(weaviate_client, COLL)
    rag.ensure_owner_property(weaviate_client, COLL)  # idempotent
    assert rag.assign_unowned(weaviate_client, COLL, "admin@x.com") == 1
    assert rag.assign_unowned(weaviate_client, COLL, "admin@x.com") == 0
    assert rag.list_documents(weaviate_client, COLL, "admin@x.com") == ["old.pdf"]
    assert rag.list_documents(weaviate_client, COLL, "someone@x.com") == []
