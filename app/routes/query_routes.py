from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from app.dependecies import rag_creator, vector_stores
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
client = genai.Client()

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    k: int = 4
    vector_store_name: str

class ChunkResponse(BaseModel):
    rank: int
    content: str
    metadata: dict
    similarity_score: Optional[float] = None


@router.post("/", response_model=List[ChunkResponse])
async def query_documents(request: QueryRequest):
    """
    Retrieve top-k similar chunks from the vector store.
    """
    try:
        if request.vector_store_name not in vector_stores:
            try:
                vector_store = rag_creator.load_existing_vector_store(
                    vector_store_name=request.vector_store_name,
                    persist_directory="./vector_stores"
                )
                vector_stores[request.vector_store_name] = vector_store
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail=f"Vector store '{request.vector_store_name}' not found")

        vector_store = vector_stores[request.vector_store_name]
        results = rag_creator.retrieve_chunks_formatted(
            vector_store=vector_store,
            query=request.query,
            k=request.k,
            include_metadata=True,
            include_scores=True
        )
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/context/")
async def query_as_context(request: QueryRequest):
    """
    Query and return chunks as a single context string for LLM.
    """
    try:
        if request.vector_store_name not in vector_stores:
            vector_store = rag_creator.load_existing_vector_store(
                vector_store_name=request.vector_store_name,
                persist_directory="./vector_stores"
            )
            vector_stores[request.vector_store_name] = vector_store

        vector_store = vector_stores[request.vector_store_name]

        context = rag_creator.retrieve_as_context(
            vector_store=vector_store,
            query=request.query,
            k=request.k
        )

        prompt = f"""
        You are a helpful AI assistant.
        Use the following context to answer the question accurately.

        Context:
        {context}

        Question:
        {request.query}

        Provide a direct, concise answer based only on the context above.
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        )

        return JSONResponse(
            status_code=200,
            content={
                "query": request.query,
                "response": response.text,
                "context_used": context,
                "chunk_count": request.k
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
