from fastapi import APIRouter
from pydantic import BaseModel
from groq import Groq
from retrieval.hybrid import hybrid_search
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list

@router.post("/query")
def query_contracts(request: QueryRequest):
    chunks = hybrid_search(request.question, n_results=3)

    context = "\n\n---\n\n".join([
        f"Source: {c['source']} (chunk {c['chunk_index']})\n{c['text']}"
        for c in chunks
    ])

    source_list = [c['source'] for c in chunks]

    prompt = f"""You are a contract analysis assistant. You have been given excerpts from the following sources: {source_list}.

Answer the question using ONLY the contract excerpts below.
After your answer:
- State which document(s) contained the relevant clause
- Flag any terms the user should verify with a lawyer
- If the excerpts are incomplete or cut off, say so explicitly
If the answer is not in the excerpts, say "I could not find this information in the provided contracts."

Contract excerpts:
{context}

Question: {request.question}

Answer:"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024
    )

    sources = [
        {
            "source": c["source"],
            "chunk_index": c["chunk_index"],
            "score": round(c["score"], 3)
        }
        for c in reversed(chunks)
    ]

    return QueryResponse(
        answer=response.choices[0].message.content,
        sources=sources
    )