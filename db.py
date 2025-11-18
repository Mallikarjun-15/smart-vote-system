import os
from datetime import datetime
from contextlib import contextmanager

from sqlalchemy import (
    Column,
    Integer,
    String,
    LargeBinary,
    DateTime,
    ForeignKey,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy import create_engine


DATABASE_URL = os.getenv("SMARTVOTE_DB_URL", "sqlite:///smartvote.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    phone = Column(String(50), unique=True, nullable=True)
    password_hash = Column(String(255), nullable=False)
    face_embedding = Column(LargeBinary, nullable=True)
    gov_id_path = Column(Text, nullable=True)
    role = Column(String(50), default="voter")
    failed_face_attempts = Column(Integer, default=0)
    last_failed_face_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    votes = relationship("Vote", back_populates="voter")


class Election(Base):
    __tablename__ = "elections"

    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)

    candidates = relationship("Candidate", back_populates="election", cascade="all, delete-orphan")
    votes = relationship("Vote", back_populates="election", cascade="all, delete-orphan")


class Candidate(Base):
    __tablename__ = "candidates"

    id = Column(Integer, primary_key=True)
    election_id = Column(Integer, ForeignKey("elections.id"), nullable=False)
    name = Column(String(255), nullable=False)
    party = Column(String(255), nullable=True)

    election = relationship("Election", back_populates="candidates")
    votes = relationship("Vote", back_populates="candidate", cascade="all, delete-orphan")

    __table_args__ = (UniqueConstraint("election_id", "name", name="uq_candidate_name_per_election"),)


class Vote(Base):
    __tablename__ = "votes"

    id = Column(Integer, primary_key=True)
    voter_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    election_id = Column(Integer, ForeignKey("elections.id"), nullable=False)
    candidate_id = Column(Integer, ForeignKey("candidates.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    voter = relationship("User", back_populates="votes")
    election = relationship("Election", back_populates="votes")
    candidate = relationship("Candidate", back_populates="votes")

    __table_args__ = (
        UniqueConstraint("voter_id", "election_id", name="uq_vote_per_voter_per_election"),
    )


def init_db():
    Base.metadata.create_all(bind=engine)
    return engine


@contextmanager
def session_scope():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

