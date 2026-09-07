import logging
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.config import settings
from app.models import Base

logger = logging.getLogger(__name__)

# Configure SQLAlchemy engine
# SQLite requires check_same_thread=False for multithreaded web applications
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db() -> None:
    """
    Initializes database tables according to SQLAlchemy ORM metadata models.
    """
    try:
        Base.metadata.create_all(bind=engine)
        logger.info(f"Database initialized successfully at {settings.DATABASE_URL}")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")

def get_db() -> Generator[Session, None, None]:
    """
    Dependency generator for acquiring a database session.
    Automatically closes session upon request/operation completion.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
