import io
from pathlib import Path

import pytest
import respx
from fastapi import UploadFile
from httpx import Response

from tolka.pipeline.fetch import (
    AudioTooLargeError,
    ForbiddenUrlError,
    fetch_url,
    save_upload,
)

# 93.184.216.34 is a public-range IP literal: no DNS lookup, passes the SSRF check,
# and respx intercepts before any real connection is attempted.
PUBLIC_URL = "http://93.184.216.34/audio/meeting.mp3"


@respx.mock
async def test_fetch_writes_file_with_suffix(tmp_path: Path):
    respx.get(PUBLIC_URL).mock(return_value=Response(200, content=b"audio-bytes"))

    dest = await fetch_url(PUBLIC_URL, dest_dir=tmp_path, max_bytes=1024, timeout_s=5)

    assert dest.parent == tmp_path
    assert dest.suffix == ".mp3"
    assert dest.read_bytes() == b"audio-bytes"


async def test_fetch_rejects_non_http_scheme(tmp_path: Path):
    with pytest.raises(ForbiddenUrlError):
        await fetch_url("ftp://example.org/a.mp3", dest_dir=tmp_path, max_bytes=10, timeout_s=5)


async def test_fetch_rejects_url_credentials(tmp_path: Path):
    with pytest.raises(ForbiddenUrlError, match="user information"):
        await fetch_url(
            "https://user:secret@example.org/audio.mp3",
            dest_dir=tmp_path,
            max_bytes=100,
            timeout_s=1,
        )


async def test_fetch_enforces_host_allowlist(tmp_path: Path):
    with pytest.raises(ForbiddenUrlError, match="allowlist"):
        await fetch_url(
            "https://example.org/audio.mp3",
            dest_dir=tmp_path,
            max_bytes=100,
            timeout_s=1,
            allowed_hosts=("media.example.org",),
        )


async def test_fetch_rejects_private_host(tmp_path: Path):
    for url in ("http://127.0.0.1/a.mp3", "http://192.168.1.10/a.mp3", "http://localhost/a.mp3"):
        with pytest.raises(ForbiddenUrlError):
            await fetch_url(url, dest_dir=tmp_path, max_bytes=10, timeout_s=5)


@respx.mock
async def test_fetch_rejects_redirect_to_private_host(tmp_path: Path):
    respx.get(PUBLIC_URL).mock(
        return_value=Response(302, headers={"location": "http://127.0.0.1/secret.mp3"})
    )
    with pytest.raises(ForbiddenUrlError):
        await fetch_url(PUBLIC_URL, dest_dir=tmp_path, max_bytes=1024, timeout_s=5)


@respx.mock
async def test_fetch_allows_private_host_when_configured(tmp_path: Path):
    url = "http://192.168.1.10/a.mp3"
    respx.get(url).mock(return_value=Response(200, content=b"ok"))

    dest = await fetch_url(url, dest_dir=tmp_path, max_bytes=1024, timeout_s=5, allow_private=True)
    assert dest.read_bytes() == b"ok"


@respx.mock
async def test_fetch_aborts_oversize_mid_stream(tmp_path: Path):
    respx.get(PUBLIC_URL).mock(return_value=Response(200, content=b"x" * 100))

    with pytest.raises(AudioTooLargeError):
        await fetch_url(PUBLIC_URL, dest_dir=tmp_path, max_bytes=50, timeout_s=5)
    assert list(tmp_path.iterdir()) == []


async def test_save_upload(tmp_path: Path):
    upload = UploadFile(io.BytesIO(b"uploaded"), filename="talk.wav")

    dest = await save_upload(upload, dest_dir=tmp_path, max_bytes=1024)
    assert dest.suffix == ".wav"
    assert dest.read_bytes() == b"uploaded"


async def test_save_upload_rejects_oversize(tmp_path: Path):
    upload = UploadFile(io.BytesIO(b"x" * 100), filename="talk.wav")

    with pytest.raises(AudioTooLargeError):
        await save_upload(upload, dest_dir=tmp_path, max_bytes=50)
    assert list(tmp_path.iterdir()) == []
