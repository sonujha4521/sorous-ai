import hashlib
import hmac
import secrets

from fastapi import HTTPException
from sqlalchemy.orm import Session

from models import User


# Active login tokens
active_tokens = {}


def hash_password(password: str) -> str:
    return hashlib.sha256(
        password.encode("utf-8")
    ).hexdigest()


def verify_password(
    plain_password: str,
    stored_password: str
) -> bool:

    hashed_password = hash_password(
        plain_password
    )

    return hmac.compare_digest(
        hashed_password,
        stored_password
    )


def create_token(user_id: int) -> str:

    token = secrets.token_urlsafe(32)

    active_tokens[token] = user_id

    return token


def logout_token(token: str):

    active_tokens.pop(
        token,
        None
    )


def get_current_user_from_token(
    token: str,
    db: Session
) -> User:

    if not token:
        raise HTTPException(
            status_code=401,
            detail="Login required"
        )


    user_id = active_tokens.get(token)


    if user_id is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token"
        )


    user = (
        db.query(User)
        .filter(
            User.id == user_id
        )
        .first()
    )


    if not user:
        active_tokens.pop(
            token,
            None
        )

        raise HTTPException(
            status_code=401,
            detail="User not found"
        )


    return user