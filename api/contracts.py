from fastapi import APIRouter
import pandas as pd

router = APIRouter()

@router.get("/contracts")
def list_contracts():
    df = pd.read_parquet("data/chunks/chunks.parquet")
    contracts = df.groupby("source").agg(
        total_chunks=("chunk_index", "count")
    ).reset_index()

    return {
        "contracts": contracts.to_dict(orient="records"),
        "total_documents": len(contracts),
        "total_chunks": len(df)
    }