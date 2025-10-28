from sqlmodel import create_engine, Session

# For SQLite (replace with your actual DB URL if needed)
DATABASE_URL = "postgresql+psycopg2://postgres:Password@localhost/postgres"

engine = create_engine(DATABASE_URL, echo=True)

def get_session():
    with Session(engine) as session:
        yield session