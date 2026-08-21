import secrets

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer(auto_error=False)


def require_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    token_clients: dict[str, str] = request.app.state.deps.settings.token_clients
    if not token_clients:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="no API tokens configured")
    client_id = None
    if credentials is not None:
        for token, candidate_client_id in token_clients.items():
            if secrets.compare_digest(credentials.credentials, token):
                client_id = candidate_client_id
    if client_id is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    request.state.client_id = client_id
    return client_id
