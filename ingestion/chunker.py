import re
from pathlib import Path
import pdfplumber

def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text.strip()

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\.{3,}', '', text)
    text = re.sub(r'\d+\s*\|\s*Page', '', text)

    # add these three:
    text = re.sub(r'https?://\S+', '', text)                    # removes URLs
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4},?\s+\d+:\d+\s+[AP]M', '', text)  # removes date-time headers
    text = re.sub(r'EX-\d+\.?\d*\s+\d+\s+\S+\.htm', '', text)  # removes exhibit file references
    text = re.sub(r'EX-[\d.]+\s+Ex\s+[\d.]+', '', text)
    text = re.sub(r'Email:\s*', '', text)
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    text = re.sub(r'\bDocument\b', '', text)
    text = re.sub(r'Attn:\s*', '', text)

    return text.strip()

def chunk_text(text, source, chunk_size=150, overlap=30):
    words = text.split()
    chunks = []
    i = 0
    chunk_index = 0
    
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words)
        
        chunks.append({
            "text": chunk,
            "source": source,
            "chunk_index": chunk_index,
            "total_chunks": -1
        })
        
        i += chunk_size - overlap
        chunk_index += 1
    
    for chunk in chunks:
        chunk["total_chunks"] = len(chunks)
    
    return chunks

def process_pdf(pdf_path):
    path = Path(pdf_path)
    raw_text = extract_text(path)
    clean = clean_text(raw_text)
    chunks = chunk_text(clean, source=path.name)
    return chunks