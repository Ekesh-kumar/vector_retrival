from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import Session
from app.models import User
from app.auth.hashes import hash_password, verify_password
from app.auth.jwt_utils import create_access_token
from app.schemas import UserCreate, UserRead, Token
from app.data_base import get_session

router = APIRouter()

@router.post("/signup")
def signup(user_create: UserCreate, session: Session = Depends(get_session)):
    # Check if user exists
    existing_user = session.exec(select(User).where(User.email == user_create.email)).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user = User(email=user_create.email, password=hash_password(user_create.password))
    try:
        session.add(user)
        session.commit()            # Save to DB
        session.refresh(user)       # Refresh to get the ID
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail="Failed to insert user")
    return user

@router.post("/signin", response_model=Token)
def signin(user_create: UserCreate, session: Session = Depends(get_session)):
    statement = select(User).where(User.email == user_create.email)
    results = session.exec(statement)
    user = results.first()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    if not verify_password(user_create.password, user.password):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}
