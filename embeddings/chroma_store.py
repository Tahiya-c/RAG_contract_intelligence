import chromadb
import pandas as pd
from embeddings.embed import load_chunks, embed_chunks

def get_chroma_client():
    return chromadb.PersistentClient(path="data/chroma")

def get_or_create_collection(client):
    return client.get_or_create_collection(
        name="contracts",
        metadata={"hnsw:space": "cosine"}
    )

def store_embeddings(df, embeddings):
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    ids = [f"{row['source']}_{row['chunk_index']}" for _, row in df.iterrows()]

    metadatas = [
        {
            "source": row["source"],
            "chunk_index": int(row["chunk_index"]),
            "total_chunks": int(row["total_chunks"])
        }
        for _, row in df.iterrows()
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=df["text"].tolist(),
        metadatas=metadatas
    )

    print(f"Stored {collection.count()} chunks in ChromaDB")
    return collection

def verify_search(collection):
    print("\nRunning test search: 'termination penalty'")
    results = collection.query(
        query_texts=["termination penalty"],
        n_results=3
    )
    for i, (doc, meta) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0]
    )):
        print(f"\nResult {i+1}: {meta['source']} chunk {meta['chunk_index']}")
        print(f"Preview: {doc[:150]}")

if __name__ == "__main__":
    df = load_chunks()
    embeddings = embed_chunks(df)
    collection = store_embeddings(df, embeddings)
    verify_search(collection)