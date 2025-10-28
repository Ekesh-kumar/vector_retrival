# app/models.py
from sqlmodel import SQLModel, Field
from typing import Optional


class ChunkMeta(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    vectorstore_name: str = Field(index=True)
    chunk_id: int
    chunk_text: str
    chunk_metadata: str  # store as JSON string