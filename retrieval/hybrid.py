from rank_bm25 import BM25Okapi
import chromadb
import pandas as pd
import numpy as np

def load_bm25_index(parquet_path="data/chunks/chunks.parquet"):
    df = pd.read_parquet(parquet_path).reset_index(drop=True)
    tokenized = [text.lower().split() for text in df["text"].tolist()]
    bm25 = BM25Okapi(tokenized)
    return bm25, df

def get_chroma_collection():
    client = chromadb.PersistentClient(path="data/chroma")
    return client.get_collection(name="contracts")

def normalize(scores):
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s == 0:
        return np.zeros_like(scores)
    return (scores - min_s) / (max_s - min_s)

def hybrid_search(question, n_results=3):
    bm25, df = load_bm25_index()
    collection = get_chroma_collection()

    tokenized_query = question.lower().split()
    bm25_scores = normalize(np.array(bm25.get_scores(tokenized_query)))

    results = collection.query(
        query_texts=[question],
        n_results=len(df),
        include=["distances", "documents", "metadatas"]
    )

    vector_map = {}
    for chunk_id, distance in zip(results["ids"][0], results["distances"][0]):
        vector_map[chunk_id] = 1 - distance

    vector_scores = np.array([
        vector_map.get(f"{row.source}_{row.chunk_index}", 0)
        for row in df.itertuples()
    ])
    vector_scores = normalize(vector_scores)

    combined_scores = 0.5 * bm25_scores + 0.5 * vector_scores
    top_indices = np.argsort(combined_scores)[::-1][:n_results]

    top_chunks = []
    for idx in top_indices:
        row = df.iloc[idx]
        top_chunks.append({
            "text": row["text"],
            "source": row["source"],
            "chunk_index": int(row["chunk_index"]),
            "total_chunks": int(row["total_chunks"]),
            "score": float(combined_scores[idx])
        })

    return top_chunks[::-1]