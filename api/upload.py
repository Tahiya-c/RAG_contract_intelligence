from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os
import json
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta
import pdfplumber
import re

router = APIRouter()

CONTRACTS_DIR = "data/sample_contracts"
DELETED_DIR = "data/deleted_contracts"
PARQUET_PATH = "data/chunks/chunks.parquet"
CHROMA_PATH = "data/chroma"
METADATA_PATH = "data/deleted_contracts/metadata.json"

os.makedirs(DELETED_DIR, exist_ok=True)


def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\bEX-\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, source, chunk_size=150, overlap=30):
    words = text.split()
    chunks = []
    start = 0
    chunk_index = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append({
            "text": " ".join(chunk_words),
            "source": source,
            "chunk_index": chunk_index,
            "total_chunks": 0
        })
        chunk_index += 1
        start += chunk_size - overlap
    for c in chunks:
        c["total_chunks"] = len(chunks)
    return chunks

def extract_chunks_from_pdf(pdf_path, filename):
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            full_text += " " + page_text
    full_text = clean_text(full_text)
    return chunk_text(full_text, filename)

def load_metadata():
    if not os.path.exists(METADATA_PATH):
        return {}
    with open(METADATA_PATH, "r") as f:
        return json.load(f)

def save_metadata(metadata):
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f)

def purge_expired(metadata):
    now = datetime.utcnow()
    to_purge = [
        name for name, ts in metadata.items()
        if now - datetime.fromisoformat(ts) > timedelta(days=7)
    ]
    for name in to_purge:
        path = os.path.join(DELETED_DIR, name)
        if os.path.exists(path):
            os.remove(path)
        del metadata[name]
    return metadata

def add_to_chroma_and_parquet(pdf_path, filename):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    chunks = extract_chunks_from_pdf(pdf_path, filename)
    if not chunks:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")

    new_df = pd.DataFrame(chunks)
    existing_df = pd.read_parquet(PARQUET_PATH)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df.to_parquet(PARQUET_PATH, index=False)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name="contracts")

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts).tolist()
    ids = [f"{c['source']}_{c['chunk_index']}" for c in chunks]
    metadatas = [{"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )
    return len(chunks)

@router.post("/upload")
async def upload_contract(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    save_path = os.path.join(CONTRACTS_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks_added = add_to_chroma_and_parquet(save_path, file.filename)
    return {"message": f"Uploaded {file.filename}", "chunks_added": chunks_added}

@router.delete("/contracts/{source_name}")
def delete_contract(source_name: str):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name="contracts")

    results = collection.get(where={"source": source_name})
    if not results["ids"]:
        raise HTTPException(status_code=404, detail=f"{source_name} not found")

    collection.delete(ids=results["ids"])

    df = pd.read_parquet(PARQUET_PATH)
    df = df[df["source"] != source_name]
    df.to_parquet(PARQUET_PATH, index=False)

    src = os.path.join(CONTRACTS_DIR, source_name)
    dst = os.path.join(DELETED_DIR, source_name)
    if os.path.exists(src):
        shutil.move(src, dst)

    metadata = load_metadata()
    metadata = purge_expired(metadata)
    metadata[source_name] = datetime.utcnow().isoformat()
    save_metadata(metadata)

    return {"message": f"Moved {source_name} to trash", "chunks_removed": len(results["ids"])}

@router.get("/deleted")
def list_deleted():
    metadata = load_metadata()
    metadata = purge_expired(metadata)
    save_metadata(metadata)
    now = datetime.utcnow()
    result = []
    for name, ts in metadata.items():
        deleted_at = datetime.fromisoformat(ts)
        days_remaining = 7 - (now - deleted_at).days
        result.append({
            "source": name,
            "deleted_at": ts,
            "days_remaining": days_remaining
        })
    return {"deleted": result}

@router.post("/restore/{source_name}")
def restore_contract(source_name: str):
    metadata = load_metadata()
    if source_name not in metadata:
        raise HTTPException(status_code=404, detail=f"{source_name} not in trash")

    src = os.path.join(DELETED_DIR, source_name)
    dst = os.path.join(CONTRACTS_DIR, source_name)
    if not os.path.exists(src):
        raise HTTPException(status_code=404, detail="PDF file no longer exists in trash")

    shutil.move(src, dst)
    chunks_added = add_to_chroma_and_parquet(dst, source_name)

    del metadata[source_name]
    save_metadata(metadata)

    return {"message": f"Restored {source_name}", "chunks_added": chunks_added}