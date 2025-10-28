from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import os
import sqlite3
from pprint import pprint
from app.dependecies import vector_stores
from app.dependecies.dependency import get_current_user
from fastapi import APIRouter, Depends
from app.models.user import User
router = APIRouter()

@router.get("/")
async def list_vector_stores(current_user: User = Depends(get_current_user)):
    """
    List all available vector stores from memory and disk.
    """
    print(f"email: {current_user.email} id: current_user.id")

    memory_stores = list(vector_stores.keys())
    db_path = "vector_stores/chroma.sqlite3"

    collections = []
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM collections")
        collections = cursor.fetchall()
        conn.close()
        pprint(collections)

    return JSONResponse(
        status_code=200,
        content={
            "loaded_in_memory": memory_stores,
            "available_on_disk": collections
        }
    )


@router.delete("/{vector_store_name}")
async def delete_vector_store(vector_store_name: str):
    """
    Delete a vector store from memory and disk.
    """
    try:
        if vector_store_name in vector_stores:
            del vector_stores[vector_store_name]

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
