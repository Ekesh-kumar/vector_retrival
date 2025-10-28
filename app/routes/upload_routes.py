from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
from app.utilities.ragHandler import RAGVectorStoreCreator

# You’ll inject these later from main
from app.dependecies import rag_creator, vector_stores

router = APIRouter()

@router.post("/document/")
async def upload_document(file: UploadFile = File(...), vector_store_name: Optional[str] = None):
    """
    Upload a PDF or TXT file and create a RAG vector store.
    """
    try:
        if not file.filename.endswith(('.pdf', '.txt')):
            raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")

        result = await rag_creator.create_rag_from_file(
            file=file,
            vector_store_name=vector_store_name,
            persist_directory="./vector_stores"
        )

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
