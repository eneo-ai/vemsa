"""HTTP Basic auth for the operator dashboard.

Open unless VEMSA_OPS_USER and VEMSA_OPS_PASSWORD are both set: the service is
meant for an internal network and the API port is loopback-bound by default.
Browsers answer the challenge natively and resend the credentials on every
same-origin fetch, so the page needs no login form."""

import hmac

from fastapi import Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

_basic = HTTPBasic(auto_error=False, realm="vemsa ops")


def require_ops_auth(
    request: Request,
    credentials: HTTPBasicCredentials | None = Depends(_basic),
) -> None:
    settings = request.app.state.deps.settings
    if settings.ops_user is None or settings.ops_password is None:
        return
    # bitwise `&` so both comparisons always run
    ok = credentials is not None and (
        hmac.compare_digest(credentials.username.encode(), settings.ops_user.encode())
        & hmac.compare_digest(credentials.password.encode(), settings.ops_password.encode())
    )
    if not ok:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="ops credentials required",
            headers={"WWW-Authenticate": 'Basic realm="vemsa ops"'},
        )


def no_store(response: Response) -> None:
    """Dashboard data is live; never let a proxy or the browser cache it."""
    response.headers["Cache-Control"] = "no-store"
