from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import config

# Setup database URL
database_url = config.DATABASE_URL

# Check if SQLite is used
is_sqlite = database_url.startswith("sqlite")

# Create engine
if is_sqlite:
    # check_same_thread=False is required for SQLite in FastAPI context
    engine = create_engine(
        database_url, connect_args={"check_same_thread": False}
    )
else:
    engine = create_engine(database_url)

# SessionLocal factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Declarative base class for models
Base = declarative_base()

# Dependency provider for FastAPI routes
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
