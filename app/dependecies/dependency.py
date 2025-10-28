from app.utilities.ragHandler import RAGVectorStoreCreator
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlmodel import Session, select
from app.auth.jwt_utils import verify_access_token
from app.models.user import User
from app.data_base import get_session

# Initialize global RAG creator and vector stores here
rag_creator = RAGVectorStoreCreator(
    chunk_size=1000,
    chunk_overlap=200
)

vector_stores = {}

# 1️⃣ Tell FastAPI where to get tokens from

security = HTTPBearer()

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    session: Session = Depends(get_session)
):
    token = credentials.credentials  # extract the token from header
    payload = verify_access_token(token)

    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )

    email = payload.get("sub")
    user = session.exec(select(User).where(User.email == email)).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    
    return user