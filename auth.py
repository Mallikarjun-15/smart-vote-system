from datetime import datetime, timedelta
from typing import Optional

import bcrypt

from db import SessionLocal, User


FAILED_FACE_LIMIT = 5
FAILED_FACE_COOLDOWN_MIN = 5


def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))
    except ValueError:
        return False


def get_user_by_email(session: SessionLocal, email: str) -> Optional[User]:
    return session.query(User).filter(User.email == email).first()


def get_user_by_id(session: SessionLocal, user_id: int) -> Optional[User]:
    return session.query(User).filter(User.id == user_id).first()


def register_user(
    session: SessionLocal,
    *,
    full_name: str,
    email: str,
    password: str,
    phone: Optional[str] = None,
    role: str = "voter",
    face_embedding_blob: Optional[bytes] = None,
    gov_id_path: Optional[str] = None,
) -> User:
    email = email.strip().lower()
    if get_user_by_email(session, email):
        raise ValueError("Email already registered.")

    if phone:
        existing_phone = session.query(User).filter(User.phone == phone).first()
        if existing_phone:
            raise ValueError("Phone already registered.")

    user = User(
        full_name=full_name.strip(),
        email=email,
        phone=phone.strip() if phone else None,
        password_hash=hash_password(password),
        face_embedding=face_embedding_blob,
        gov_id_path=gov_id_path,
        role=role,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def authenticate_user(session: SessionLocal, email: str, password: str) -> Optional[User]:
    user = get_user_by_email(session, email.strip().lower())
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


def record_failed_face_attempt(session: SessionLocal, user: User) -> None:
    user.failed_face_attempts = (user.failed_face_attempts or 0) + 1
    user.last_failed_face_at = datetime.utcnow()
    session.add(user)
    session.commit()


def reset_failed_face_attempts(session: SessionLocal, user: User) -> None:
    user.failed_face_attempts = 0
    user.last_failed_face_at = None
    session.add(user)
    session.commit()


def can_attempt_face_verification(user: User) -> bool:
    attempts = user.failed_face_attempts or 0
    if attempts < FAILED_FACE_LIMIT:
        return True
    if not user.last_failed_face_at:
        return True
    cooldown = user.last_failed_face_at + timedelta(minutes=FAILED_FACE_COOLDOWN_MIN)
    return datetime.utcnow() >= cooldown

