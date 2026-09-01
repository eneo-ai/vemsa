import asyncio
import ipaddress
import socket
from collections.abc import Collection
from urllib.parse import urlparse


class ForbiddenUrlError(ValueError):
    pass


async def validate_outbound_url(
    url: str,
    *,
    allow_private: bool,
    allowed_hosts: Collection[str] = (),
    require_https: bool = False,
) -> None:
    """Validate an outbound URL before use.

    Host allowlists are the strongest application-level control. Public-address
    validation is defense in depth and must be paired with production egress rules
    because a hostname can change between validation and connection.
    """
    parsed = urlparse(url)
    allowed_schemes = {"https"} if require_https else {"http", "https"}
    if parsed.scheme not in allowed_schemes:
        raise ForbiddenUrlError(f"unsupported URL scheme {parsed.scheme!r}")
    if parsed.username is not None or parsed.password is not None:
        raise ForbiddenUrlError("URL user information is not allowed")
    host = parsed.hostname
    if not host:
        raise ForbiddenUrlError("URL has no host")
    normalized_allowlist = {item.casefold().rstrip(".") for item in allowed_hosts}
    if normalized_allowlist and host.casefold().rstrip(".") not in normalized_allowlist:
        raise ForbiddenUrlError(f"host {host!r} is not in the outbound allowlist")
    if allow_private:
        return
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ForbiddenUrlError(f"cannot resolve host {host!r}") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global:
            raise ForbiddenUrlError(f"host {host!r} resolves to non-public address {address}")
