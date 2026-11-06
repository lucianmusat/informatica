"""Document ingestion and retrieval.

Retrieval deliberately bypasses WeaviateVectorStore.similarity_search_with_score():
that method runs a *hybrid* (BM25 + vector) query and hands back Weaviate's fused
score, which is normalised across each result set — the best hit scores near the top
of the range whether or not it is actually relevant. Useless as an absolute
"is this worth showing the model?" signal. A plain near_vector query gives a real
cosine distance instead, which is comparable across queries and can be thresholded.

Documents are private to the user who uploaded them. Every chunk carries an
``owner`` (the uploader's email) and every read, list and delete filters on it.
``owner`` uses FIELD tokenization so the filter is an exact match: with the default
WORD tokenization, filtering on "bob@gmail.com" would also match
"x.bob@gmail.com", because Weaviate compares the word tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter
from weaviate.classes.aggregate import GroupByAggregate
from weaviate.classes.config import DataType, Property, Tokenization
from weaviate.classes.query import Filter, MetadataQuery

from config import CHUNK_OVERLAP, CHUNK_SIZE


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk plus how far it sat from the query."""

    text: str
    file_name: str
    distance: float

    @property
    def similarity(self) -> float:
        """Cosine distance expressed as a 0-1 similarity, for display."""
        return max(0.0, 1.0 - self.distance / 2.0)


def split_text(text: str) -> list[str]:
    """Chunk a document.

    Recursive splitting rather than plain CharacterTextSplitter: the latter only
    breaks on its single separator, so a PDF page with no newlines came out as one
    enormous chunk that blew past chunk_size.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [c for c in splitter.split_text(text) if c.strip()]


OWNER = "owner"

def collection_properties() -> list[Property]:
    """Schema for a new document collection."""
    return [
        Property(name="text", data_type=DataType.TEXT),
        Property(name="fileName", data_type=DataType.TEXT),
        Property(name=OWNER, data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
    ]


def ensure_owner_property(client, class_name: str) -> None:
    """Add ``owner`` to a collection created before documents were per-user.

    Refuses to run on an ``owner`` that already exists with another tokenization,
    because that filter would not be an exact match (see module docstring).
    """
    collection = client.collections.get(class_name)
    existing = {p.name: p for p in collection.config.get().properties}
    if OWNER not in existing:
        collection.config.add_property(
            Property(name=OWNER, data_type=DataType.TEXT, tokenization=Tokenization.FIELD)
        )
    elif existing[OWNER].tokenization != Tokenization.FIELD:
        raise RuntimeError(
            f"{class_name}.{OWNER} has tokenization {existing[OWNER].tokenization}, "
            "expected FIELD; per-user filtering would not be exact."
        )


def assign_unowned(client, class_name: str, owner: str) -> int:
    """Give chunks stored before per-user documents to ``owner``. Returns count."""
    collection = client.collections.get(class_name)
    unowned = [
        item.uuid
        for item in collection.iterator(return_properties=[OWNER])
        if not (item.properties or {}).get(OWNER)
    ]
    for uuid in unowned:
        collection.data.update(uuid=uuid, properties={OWNER: owner})
    return len(unowned)


def _owned_by(owner: str):
    return Filter.by_property(OWNER).equal(owner)


def store_document(vectorstore, file_name: str, text: str, owner: str, *, client=None,
                   class_name: str | None = None) -> int:
    """Embed and persist one document for ``owner``. Returns chunks stored.

    With ``client`` and ``class_name``, an earlier upload of the same file by
    the same owner is replaced: the old chunks are deleted only after the new
    ones are stored, so the document never disappears in between. Another
    user's file of the same name is never touched.
    """
    chunks = split_text(text)
    if not chunks:
        return 0
    old = _ids_of(client.collections.get(class_name), file_name, owner) if client else []
    vectorstore.add_texts(
        chunks, metadatas=[{"fileName": file_name, OWNER: owner} for _ in chunks]
    )
    if old:
        client.collections.get(class_name).data.delete_many(where=Filter.by_id().contains_any(old))
    return len(chunks)


def retrieve(
    client, embeddings, class_name: str, query: str, k: int, *, owner: str | None,
    files: list[str] | None = None,
) -> list[Hit]:
    """Nearest chunks by cosine distance, closest first.

    ``owner`` is required on purpose: the app always passes the signed-in user.
    ``None`` (no filter) is only for single-tenant collections such as the
    calibration tool's.

    ``files`` limits the search to those of the owner's files. fileName is
    WORD-tokenized, so the filter can also let through similarly named files
    ("a.pdf" matches "a b.pdf"); more candidates are fetched and the names are
    checked exactly here.
    """
    collection = client.collections.get(class_name)
    vector = embeddings.embed_query(query)

    filters = _owned_by(owner) if owner is not None else None
    if files:
        by_name = Filter.any_of([Filter.by_property("fileName").equal(f) for f in files])
        filters = by_name if filters is None else filters & by_name

    result = collection.query.near_vector(
        near_vector=vector,
        limit=k * 5 if files else k,
        filters=filters,
        return_metadata=MetadataQuery(distance=True),
    )

    hits: list[Hit] = []
    for obj in result.objects:
        props = obj.properties or {}
        text = props.get("text") or props.get("body") or ""
        if not text or files and props.get("fileName") not in files:
            continue
        distance = obj.metadata.distance
        hits.append(
            Hit(
                text=text,
                file_name=props.get("fileName") or "unknown",
                # A missing distance shouldn't silently look like a perfect match.
                distance=float(distance) if distance is not None else 2.0,
            )
        )
    return hits[:k]


def list_documents(client, class_name: str, owner: str) -> list[str]:
    """Distinct file names ``owner`` has stored."""
    collection = client.collections.get(class_name)
    result = collection.aggregate.over_all(
        group_by=GroupByAggregate(prop="fileName", limit=10_000),
        filters=_owned_by(owner),
    )
    return sorted({g.grouped_by.value for g in result.groups if g.grouped_by.value})


def delete_document(client, class_name: str, file_name: str, owner: str) -> int:
    """Drop every chunk of one of ``owner``'s files. Returns objects deleted.

    fileName is WORD-tokenized, so filtering on "a.pdf" also matches "a b.pdf".
    The filter only narrows the candidates; the exact name is checked here and the
    delete goes by id.
    """
    collection = client.collections.get(class_name)
    ids = _ids_of(collection, file_name, owner)
    if not ids:
        return 0
    result = collection.data.delete_many(where=Filter.by_id().contains_any(ids))
    return result.matches


def _ids_of(collection, file_name: str, owner: str) -> list:
    """Ids of every chunk of exactly ``file_name`` owned by ``owner``."""
    candidates = collection.query.fetch_objects(
        filters=_owned_by(owner) & Filter.by_property("fileName").equal(file_name),
        return_properties=["fileName"],
        limit=10_000,
    )
    return [
        o.uuid for o in candidates.objects
        if (o.properties or {}).get("fileName") == file_name
    ]
