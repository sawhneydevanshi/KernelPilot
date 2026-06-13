import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from backend.rag.chunker import DocChunk, load_chunks

load_dotenv()


CHROMA_DB_PATH = "data/chroma"
COLLECTION_NAME = "pytorch_docs"


def get_collection(reset: bool = False):
    """
    Get or create the ChromaDB collection.
    reset=True clears and rebuilds the collection from scratch.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY not set in .env file")

  
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_key,
        model_name="text-embedding-3-small",
    )
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}, 
    )

    return collection


def index_chunks(chunks: list[DocChunk], reset: bool = True) -> int:
    """
    Embed all chunks and store in ChromaDB.
    Returns the number of chunks indexed.
    """
    collection = get_collection(reset=reset)

  
    if not reset and collection.count() > 0:
        print(f"Collection already has {collection.count()} chunks. Skipping indexing.")
        return collection.count()

    print(f"Indexing {len(chunks)} chunks into ChromaDB...")

   
    ids = [chunk.chunk_id for chunk in chunks]
    documents = [chunk.text for chunk in chunks]
    metadatas = [
        {
            "source_url": chunk.source_url,
            "page_title": chunk.page_title,
            "token_count": chunk.token_count,
        }
        for chunk in chunks
    ]

    
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch_ids = ids[i:i + batch_size]
        batch_docs = documents[i:i + batch_size]
        batch_meta = metadatas[i:i + batch_size]

        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta,
        )
        print(f"  Indexed batch {i//batch_size + 1} ({len(batch_ids)} chunks)")

    print(f"Done. {collection.count()} chunks in collection.")
    return collection.count()


def retrieve(
    query: str,
    top_k: int = 3,
) -> list[dict]:
    """
    Retrieve the top_k most relevant chunks for a query.

    Returns a list of dicts with:
        text, source_url, page_title, relevance_score
    """
    collection = get_collection(reset=False)

    if collection.count() == 0:
        raise ValueError(
            "ChromaDB collection is empty. Run index_chunks() first."
        )

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    
    retrieved = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, distance in zip(documents, metadatas, distances):
       
        similarity = 1 - distance
        retrieved.append({
            "text": doc,
            "source_url": meta["source_url"],
            "page_title": meta["page_title"],
            "relevance_score": round(similarity, 4),
        })

  
    retrieved.sort(key=lambda x: x["relevance_score"], reverse=True)
    return retrieved