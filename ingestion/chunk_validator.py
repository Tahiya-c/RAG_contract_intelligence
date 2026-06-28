from tokenizers import Tokenizer
from ingestion.chunker import process_pdf
from pathlib import Path

tokenizer = Tokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def validate_chunks(chunks):
    warnings = 0
    for chunk in chunks:
        tokens = tokenizer.encode(chunk["text"]).ids
        if len(tokens) > 256:
            print(f"WARNING: {chunk['source']} chunk {chunk['chunk_index']} has {len(tokens)} tokens — will be truncated")
            warnings += 1
    return warnings

def validate_all(contracts_folder="data/sample_contracts"):
    folder = Path(contracts_folder)
    pdfs = list(folder.glob("*.pdf"))
    total_chunks = 0
    total_warnings = 0

    for pdf in pdfs:
        chunks = process_pdf(pdf)
        warnings = validate_chunks(chunks)
        total_chunks += len(chunks)
        total_warnings += warnings
        status = "OK" if warnings == 0 else f"{warnings} warnings"
        print(f"{pdf.name}: {len(chunks)} chunks — {status}")

    print(f"\nTotal: {total_chunks} chunks, {total_warnings} truncation warnings")

if __name__ == "__main__":
    validate_all()