import secrets

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer(auto_error=False)


def require_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    tokens: list[str] = request.app.state.deps.settings.api_tokens
    if not tokens:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="no API tokens configured")
    if credentials is None or not any(
        secrets.compare_digest(credentials.credentials, token) for token in tokens
    ):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
