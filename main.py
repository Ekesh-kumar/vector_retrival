from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from ragHandler import RAGVectorStoreCreator
import os
from google import genai
from dotenv import load_dotenv
import sqlite3
from pprint import pprint
load_dotenv()
client = genai.Client()

app = FastAPI()


# Initialize RAG creator (global or dependency injection)
rag_creator = RAGVectorStoreCreator(
    chunk_size=1000,
    chunk_overlap=200
)
# Store vector stores in memory (or use a database in production)
vector_stores = {}

class QueryRequest(BaseModel):
    query: str
    k: int = 4
    vector_store_name: str


class ChunkResponse(BaseModel):
    rank: int
    content: str
    metadata: dict
    similarity_score: Optional[float] = None


@app.post("/upload-document/")
async def upload_document(
    file: UploadFile = File(...),
    vector_store_name: Optional[str] = None
):
    """
    Upload a PDF or text file and create a RAG vector store.
    """
    try:
        # Validate file type
        if not file.filename.endswith(('.pdf', '.txt')):
            raise HTTPException(
                status_code=400,
                detail="Only PDF and TXT files are supported"
            )
        
        # Create RAG from uploaded file
        result = await rag_creator.create_rag_from_file(
            file=file,
            vector_store_name=vector_store_name,
            persist_directory="./vector_stores"
        )
        
        # Store vector store in memory
        vector_stores[result['vector_store_name']] = result['vector_store']
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Document processed successfully",
                "vector_store_name": result['vector_store_name'],
                "filename": result['filename'],
                "document_count": result['document_count'],
                "chunk_count": result['chunk_count']
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/", response_model=List[ChunkResponse])
async def query_documents(request: QueryRequest):
    """
    Query the vector store and retrieve relevant chunks.
    """
    try:
        # Check if vector store exists
        if request.vector_store_name not in vector_stores:
            # Try to load from disk
            try:
                vector_store = rag_creator.load_existing_vector_store(
                    vector_store_name=request.vector_store_name,
                    persist_directory="./vector_stores"
                )
                vector_stores[request.vector_store_name] = vector_store
            except FileNotFoundError:
                raise HTTPException(
                    status_code=404,
                    detail=f"Vector store '{request.vector_store_name}' not found"
                )
        
        vector_store = vector_stores[request.vector_store_name]
        
        # Retrieve chunks
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


@app.post("/query-context/")
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
        
        # Retrieve as context string
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
                "response":response.text,
                "context_used": context,
                "chunk_count": request.k
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vector-stores/")
async def list_vector_stores():
    """
    List all available vector stores.
    """
    # List from memory
    memory_stores = list(vector_stores.keys())
    
    # List from disk
    disk_stores = []
    vector_store_dir = "./vector_stores"


    # Path to your Chroma DB
    db_path = "vector_stores/chroma.sqlite3"

    # Connect
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Fetch collections
    cursor.execute("SELECT id, name FROM collections")
    collections = cursor.fetchall()

    pprint(collections)

    conn.close()
        
    return JSONResponse(
        status_code=200,
        content={
            "loaded_in_memory": memory_stores,
            "available_on_disk": collections
        }
    )


@app.delete("/vector-store/{vector_store_name}")
async def delete_vector_store(vector_store_name: str):
    """
    Delete a vector store from memory and disk.
    """
    try:
        # Remove from memory
        if vector_store_name in vector_stores:
            del vector_stores[vector_store_name]
        
        # Remove from disk
        vector_store_path = os.path.join("./vector_stores", vector_store_name)
        if os.path.exists(vector_store_path):
            import shutil
            shutil.rmtree(vector_store_path)
        
        return JSONResponse(
            status_code=200,
            content={"message": f"Vector store '{vector_store_name}' deleted"}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
