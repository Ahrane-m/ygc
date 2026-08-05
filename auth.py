"""
Auth (Phase 4)
=========================================
Every route (except /health) requires two headers:

    Authorization: Bearer <jwt>
    X-User-Id: <user_id>

The JWT is verified locally (HS256, JWT_SECRET) — no round-trip to the
`users` collection needed, since the token is the source of truth. The
X-User-Id header is cross-checked against the user id claim inside the
token as a consistency check (catches a client sending mismatched
credentials), not as a separate trust source.

Env:
    JWT_SECRET   the same secret the token issuer signs with
"""

import os
from typing import Optional

import jwt
from fastapi import Header, HTTPException

JWT_ALGORITHMS = ["HS256"]

# Different auth stacks name the user-id claim differently; accept any of
# the common ones rather than hard-coding one.
USER_ID_CLAIMS = ("user_id", "userId", "id", "_id", "sub")


def get_current_user(
    authorization: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
) -> str:
    """FastAPI dependency: validates the access token and returns the
    authenticated user_id. Raises 401 for any missing/invalid/mismatched
    credential, 500 if the server itself is missing JWT_SECRET."""
    if not authorization or not authorization.strip().lower().startswith("bearer "):
        raise HTTPException(401, "Missing or malformed Authorization header (expected 'Bearer <token>').")
    if not x_user_id or not x_user_id.strip():
        raise HTTPException(401, "Missing X-User-Id header.")

    token = authorization.split(" ", 1)[1].strip()

    secret = os.environ.get("JWT_SECRET")
    if not secret:
        raise HTTPException(500, "Server misconfigured: JWT_SECRET is not set.")

    try:
        payload = jwt.decode(token, secret, algorithms=JWT_ALGORITHMS)
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Access token has expired.")
    except jwt.InvalidTokenError as e:
        raise HTTPException(401, f"Invalid access token: {e}")

    token_user_id = None
    for claim in USER_ID_CLAIMS:
        if claim in payload:
            token_user_id = str(payload[claim])
            break
    if token_user_id is None:
        raise HTTPException(401, "Access token does not contain a recognizable user id claim.")

    if token_user_id != x_user_id.strip():
        raise HTTPException(401, "X-User-Id header does not match the authenticated access token.")

    return token_user_id
