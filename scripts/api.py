from typing import Literal, Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag_core import answer_question

app = FastAPI()

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(3, ge=1, le=10)
    discipline_filter: Literal["all", "muay-thai", "jiu-jitsu"] = "all"
    show_sources: bool = True
    temperature: float = Field(0.2, ge=0.0, le=1.0)

class QueryResponse(BaseModel):
    answer: str
    sources: str = ""

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    answer, sources = answer_question(
        question=req.question,
        top_k=req.top_k,
        discipline_filter=req.discipline_filter,
        show_sources=req.show_sources,
        temperature=req.temperature,
    )
    return {"answer": answer, "sources": sources}