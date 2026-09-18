"""Calibrate RETRIEVAL_MAX_DISTANCE.

"Auto" mode injects a chunk only when its cosine distance to the question is at or
below the threshold. Too high and every casual message drags in irrelevant excerpts;
too low and real document questions go unanswered. This script ingests a sample
document, fires on-topic and off-topic questions at it, and prints the distances so
the gap between the two groups is visible.

    docker --context default compose exec app python tools/calibrate_retrieval.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import weaviate  # noqa: E402
from langchain_ollama import OllamaEmbeddings  # noqa: E402
from langchain_weaviate.vectorstores import WeaviateVectorStore  # noqa: E402
from weaviate.classes.config import DataType, Property  # noqa: E402

import rag  # noqa: E402
from config import (  # noqa: E402
    EMBEDDER_MODEL,
    OLLAMA_URL,
    RETRIEVAL_MAX_DISTANCE,
    WEAVIATE_GRPC_PORT,
    WEAVIATE_HTTP_PORT,
    WEAVIATE_URL,
)

CALIBRATION_CLASS = "RetrievalCalibration"

SAMPLE_DOC = """
Espresso Machine Model X900 - Service Manual

Chapter 4: Boiler Descaling
The boiler must be descaled every 300 brewing cycles or every three months,
whichever comes first. Use only food-grade citric acid solution at a 5% concentration.
Never use vinegar, as the acetic acid degrades the silicone gaskets.
Run three full tank cycles, then two clean water cycles to flush residue.

Chapter 5: Group Head Calibration
Target brew pressure is 9 bar at the group head. Adjust via the overpressure valve
located beneath the drip tray. A quarter turn clockwise raises pressure by roughly
0.4 bar. Verify with a portafilter gauge at operating temperature, never cold.

Chapter 6: Error Codes
E01 indicates a boiler temperature sensor fault. E02 indicates pump overcurrent,
usually caused by a blocked grinder burr feeding too fine a grind.
E07 means the water reservoir float switch has failed.
"""

ON_TOPIC = [
    "How often should I descale the boiler?",
    "What does error code E02 mean?",
    "What brew pressure should the group head be set to?",
    "Can I use vinegar to clean the machine?",
]

OFF_TOPIC = [
    "Write me a Python function that reverses a linked list",
    "What's the capital of Portugal?",
    "Explain the difference between TCP and UDP",
    "Hey, how are you doing today?",
    "Give me a recipe for banana bread",
]


def main() -> int:
    host = WEAVIATE_URL.replace("http://", "").replace("https://", "").strip("/")
    client = weaviate.connect_to_custom(
        http_host=host,
        http_port=WEAVIATE_HTTP_PORT,
        http_secure=False,
        grpc_host=host,
        grpc_port=WEAVIATE_GRPC_PORT,
        grpc_secure=False,
    )

    try:
        if client.collections.exists(CALIBRATION_CLASS):
            client.collections.delete(CALIBRATION_CLASS)
        client.collections.create(
            CALIBRATION_CLASS,
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="fileName", data_type=DataType.TEXT),
            ],
        )

        embeddings = OllamaEmbeddings(base_url=OLLAMA_URL, model=EMBEDDER_MODEL)
        store = WeaviateVectorStore(
            client=client,
            index_name=CALIBRATION_CLASS,
            text_key="text",
            embedding=embeddings,
        )

        chunks = rag.store_document(store, "x900_service_manual.pdf", SAMPLE_DOC)
        print(f"Indexed {chunks} chunks from the sample manual.\n")

        def probe(label: str, questions: list[str]) -> list[float]:
            print(f"--- {label} " + "-" * (58 - len(label)))
            best: list[float] = []
            for question in questions:
                hits = rag.retrieve(client, embeddings, CALIBRATION_CLASS, question, 4)
                top = hits[0].distance if hits else 2.0
                best.append(top)
                flag = "RETRIEVES" if top <= RETRIEVAL_MAX_DISTANCE else "skips    "
                print(f"  {top:.3f}  {flag}  {question[:52]}")
            print()
            return best

        on = probe("on-topic (should retrieve)", ON_TOPIC)
        off = probe("off-topic (should skip)", OFF_TOPIC)

        print("=" * 64)
        print(f"  on-topic  worst (highest) distance : {max(on):.3f}")
        print(f"  off-topic best (lowest)  distance : {min(off):.3f}")
        print(f"  current RETRIEVAL_MAX_DISTANCE    : {RETRIEVAL_MAX_DISTANCE:.3f}")

        if max(on) < min(off):
            midpoint = (max(on) + min(off)) / 2
            print(f"  clean separation — suggested threshold: {midpoint:.3f}")
            ok = max(on) <= RETRIEVAL_MAX_DISTANCE < min(off)
            print(f"  current threshold classifies all probes correctly: {ok}")
        else:
            print("  WARNING: the two groups overlap; no threshold separates them.")
        print("=" * 64)
        return 0
    finally:
        if client.collections.exists(CALIBRATION_CLASS):
            client.collections.delete(CALIBRATION_CLASS)
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
