from sentence_transformers import SentenceTransformer
import pandas as pd
from pathlib import Path

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def load_chunks(parquet_path="data/chunks/chunks.parquet"):
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} chunks from {parquet_path}")
    return df

def embed_chunks(df):
    texts = df["text"].tolist()
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)
    print(f"Done. Each embedding has {embeddings.shape[1]} dimensions")
    return embeddings

if __name__ == "__main__":
    df = load_chunks()
    embeddings = embed_chunks(df)
    print(f"\nEmbedding matrix shape: {embeddings.shape}")
    print("Ready to store in ChromaDB")