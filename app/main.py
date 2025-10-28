from fastapi import FastAPI
from app.routes import upload_routes, query_routes, vectorstore_routes
from app.utilities.ragHandler import RAGVectorStoreCreator
from app.routes import auth_routes

# Initialize FastAPI app
app = FastAPI(title="RAG API Service")


# Initialize shared resources
rag_creator = RAGVectorStoreCreator(chunk_size=1000, chunk_overlap=200)
vector_stores = {}

# Include routers and pass shared dependencies
app.include_router(upload_routes.router, prefix="/upload", tags=["Upload"])
app.include_router(query_routes.router, prefix="/query", tags=["Query"])
app.include_router(vectorstore_routes.router, prefix="/vector-store", tags=["Vector Stores"])
app.include_router(auth_routes.router, prefix="/auth", tags=["auth"])
