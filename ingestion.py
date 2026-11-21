"""Load mediumblog1.txt, split it, embed its chunks, and upload them to Pinecone.

Run this before main.py. An existing index is left untouched unless you explicitly
pass --reset-index, which deletes and recreates it. Use a dedicated demo index.
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec


def main(reset_index=False):
    """Prepare the local text and populate a new or explicitly reset demo index.

    The index dimension and embedding model must match the query side in main.py.
    Rebuilding after edits replaces the old source instead of adding duplicates.
    """
    load_dotenv(Path(__file__).resolve().with_name(".env"))
    required = ("GOOGLE_API_KEY", "PINECONE_API_KEY", "INDEX_NAME")
    missing = [name for name in required if not os.getenv(name, "").strip()]
    if missing:
        raise SystemExit(
            f"Missing configuration: {', '.join(missing)}. "
            "Copy .env.example to .env and fill in your credentials."
        )

    # Pathlib works across operating systems. Plain UTF-8 text needs no external
    # document parser, spaCy model, or system-specific file path.
    source_path = Path(__file__).resolve().with_name("mediumblog1.txt")
    try:
        source_text = source_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise SystemExit(
            "mediumblog1.txt was not found. Restore the included example file "
            "from this repository next to ingestion.py."
        ) from None
    if not source_text.strip():
        raise SystemExit("mediumblog1.txt is empty. Add text before running ingestion.")
    documents = [
        Document(page_content=source_text, metadata={"source": source_path.name})
    ]

    # Target 1,000 characters (not tokens), with no overlap. Long paragraphs
    # without the default blank-line separator can exceed this target.
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {source_path.name}.")

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.environ["INDEX_NAME"]
    if pc.has_index(index_name):
        if not reset_index:
            raise SystemExit(
                f"Index '{index_name}' already exists and was left unchanged. "
                "Choose a new INDEX_NAME or run with --reset-index to delete "
                "and rebuild this demo index."
            )
        print(f"Deleting demo index '{index_name}'...")
        # Wait for deletion to finish before recreating the same index name.
        pc.delete_index(index_name, timeout=60)

    # Create a dense index using cosine similarity. A positive timeout waits for
    # the index to be ready before uploading, for up to 60 seconds.
    pc.create_index(
        name=index_name,
        vector_type="dense",
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        timeout=60,
    )

    # Turn each chunk into 1,536 numbers. main.py must use this same model/size.
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        output_dimensionality=1536,
    )
    # Upload the vectors, text, and metadata. Retrieval returns that stored text
    # so the chat model can use it as context when answering questions.
    PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name)
    print(f"Uploaded {len(chunks)} chunks to '{index_name}'.")
    print("Run main.py next. New records may take a few seconds to become searchable.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset-index",
        action="store_true",
        help="Delete and recreate INDEX_NAME, removing everything already in it.",
    )
    main(reset_index=parser.parse_args().reset_index)
