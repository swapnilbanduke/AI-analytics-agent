"""
Simple API key authentication for the FastAPI server.
"""

import os

from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer(auto_error=False)

API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN", "")


async def verify_token(
    credentials: HTTPAuthorizationCredentials | None = Security(security),
) -> bool:
    """Verify bearer token. Returns True if valid or auth is disabled."""
    # No auth in dev mode (token not set)
    if not API_AUTH_TOKEN:
        return True

    if not credentials:
        raise HTTPException(status_code=401, detail="Missing authorization header.")

    if credentials.credentials != API_AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication token.")

    return True
