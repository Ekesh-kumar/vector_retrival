from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Optional
from sqlmodel import Session
from app.data_base import get_session
from app.dependecies import rag_creator, vector_stores
import json

router = APIRouter()

@router.post("/document/")
async def upload_document(
    file: UploadFile = File(...),
    vector_store_name: Optional[str] = None,
    session: Session = Depends(get_session)
):
    """
    Upload a document, create a RAG vector store, and store per-chunk metadata.
    """
    try:
        if not file.filename.endswith(('.pdf', '.txt')):
            raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")

        # Create vector store using your RAG handler
        result = await rag_creator.create_rag_from_file(
            file=file,
            session=session,
            vector_store_name=vector_store_name,
            persist_directory="./vector_stores",
          
        )

        # Save the vector store in memory
        vector_stores[result["vector_store_name"]] = result["vector_store"]

        # === NEW: store chunk-level metadata ===
        

        return JSONResponse(
            status_code=200,
            content={
                "message": "Document processed and chunks saved successfully",
                "vector_store_name": result["vector_store_name"],
                "filename": result["filename"],
                "document_count": result["document_count"]
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
