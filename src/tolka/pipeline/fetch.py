from pathlib import Path
from urllib.parse import urljoin, urlparse
from uuid import uuid4

import httpx
from starlette.datastructures import UploadFile

from tolka.security import ForbiddenUrlError, validate_outbound_url

_CHUNK_SIZE = 1024 * 1024
_MAX_REDIRECTS = 5
_REDIRECT_CODES = {301, 302, 303, 307, 308}


class AudioTooLargeError(Exception):
    pass


def _dest_path(dest_dir: Path, name_hint: str | None) -> Path:
    suffix = Path(name_hint).suffix if name_hint else ""
    if not suffix or len(suffix) > 8:
        suffix = ".audio"
    dest_dir.mkdir(parents=True, exist_ok=True)
    return dest_dir / f"{uuid4().hex}{suffix}"


async def _write_stream(response: httpx.Response, dest: Path, max_bytes: int) -> None:
    received = 0
    try:
        with dest.open("wb") as out:
            async for chunk in response.aiter_bytes(_CHUNK_SIZE):
                received += len(chunk)
                if received > max_bytes:
                    raise AudioTooLargeError(f"download exceeds {max_bytes} bytes")
                out.write(chunk)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise


async def fetch_url(
    url: str,
    *,
    dest_dir: Path,
    max_bytes: int,
    timeout_s: float,
    allow_private: bool = False,
    allowed_hosts: tuple[str, ...] = (),
) -> Path:
    """Stream a remote audio file to dest_dir, enforcing scheme, host, and size limits.

    Redirects are followed manually so every hop gets the same private-address check.
    """
    async with httpx.AsyncClient(follow_redirects=False, timeout=timeout_s) as client:
        for _ in range(_MAX_REDIRECTS + 1):
            await validate_outbound_url(
                url,
                allow_private=allow_private,
                allowed_hosts=allowed_hosts,
            )
            parsed = urlparse(url)

            request = client.build_request("GET", url)
            response = await client.send(request, stream=True)
            try:
                if response.status_code in _REDIRECT_CODES:
                    location = response.headers.get("location")
                    if not location:
                        raise ForbiddenUrlError("redirect without Location header")
                    url = urljoin(url, location)
                    continue
                response.raise_for_status()
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > max_bytes:
                    raise AudioTooLargeError(f"source is larger than {max_bytes} bytes")
                dest = _dest_path(dest_dir, parsed.path)
                await _write_stream(response, dest, max_bytes)
                return dest
            finally:
                await response.aclose()
    raise ForbiddenUrlError(f"too many redirects fetching {url!r}")


async def save_upload(upload: UploadFile, *, dest_dir: Path, max_bytes: int) -> Path:
    dest = _dest_path(dest_dir, upload.filename)
    received = 0
    try:
        with dest.open("wb") as out:
            while chunk := await upload.read(_CHUNK_SIZE):
                received += len(chunk)
                if received > max_bytes:
                    raise AudioTooLargeError(f"upload exceeds {max_bytes} bytes")
                out.write(chunk)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    return dest
