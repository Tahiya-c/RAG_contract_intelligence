from ingestion.chunker import process_pdf

chunks = process_pdf('data/sample_contracts/farmer_brothers_board_agreement_2023.pdf')
print(f'Total chunks: {len(chunks)}')
print(f'First chunk: {chunks[0]["text"][:300]}')
print(f'Metadata: {chunks[0]["source"]}, chunk {chunks[0]["chunk_index"]} of {chunks[0]["total_chunks"]}')