from fastapi import FastAPI
from api.query import router as query_router
from api.contracts import router as contracts_router
from api.upload import router as upload_router

app = FastAPI(title="Contract Intelligence API")

app.include_router(query_router)
app.include_router(contracts_router)
app.include_router(upload_router)

@app.get("/health")
def health():
    return {"status": "ok"}
    
    