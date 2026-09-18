"""Document ingestion and retrieval.

Retrieval deliberately bypasses WeaviateVectorStore.similarity_search_with_score():
that method runs a *hybrid* (BM25 + vector) query and hands back Weaviate's fused
score, which is normalised across each result set — the best hit scores near the top
of the range whether or not it is actually relevant. Useless as an absolute
"is this worth showing the model?" signal. A plain near_vector query gives a real
cosine distance instead, which is comparable across queries and can be thresholded.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter
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


def store_document(vectorstore, file_name: str, text: str) -> int:
    """Embed and persist one document. Returns the number of chunks stored."""
    chunks = split_text(text)
    if not chunks:
        return 0
    vectorstore.add_texts(chunks, metadatas=[{"fileName": file_name} for _ in chunks])
    return len(chunks)


def retrieve(client, embeddings, class_name: str, query: str, k: int) -> list[Hit]:
    """Nearest chunks by cosine distance, closest first."""
    collection = client.collections.get(class_name)
    vector = embeddings.embed_query(query)

    result = collection.query.near_vector(
        near_vector=vector,
        limit=k,
        return_metadata=MetadataQuery(distance=True),
    )

    hits: list[Hit] = []
    for obj in result.objects:
        props = obj.properties or {}
        text = props.get("text") or props.get("body") or ""
        if not text:
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
    return hits


def list_documents(client, class_name: str) -> list[str]:
    """Distinct file names currently in the store."""
    collection = client.collections.get(class_name)
    names: set[str] = set()
    for item in collection.iterator(return_properties=["fileName"]):
        name = (item.properties or {}).get("fileName")
        if name:
            names.add(name)
    return sorted(names)


def delete_document(client, class_name: str, file_name: str) -> int:
    """Drop every chunk belonging to one file. Returns objects deleted."""
    collection = client.collections.get(class_name)
    result = collection.data.delete_many(
        where=Filter.by_property("fileName").equal(file_name)
    )
    return result.matches
