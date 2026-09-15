"""Replay a generated synthetic meeting through an explicitly selected Vemsa API.

Uses the service's real decoding/alignment/attribution path. No endpoint is assumed,
no credentials are saved, and only hash-matched generator-v2 fixtures are accepted.
"""

import argparse
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from vemsa.jobs.models import JobRequest, TranscriptionResult


def load_fixture(audio: Path, truth_path: Path):
    truth = json.loads(truth_path.read_text())
    digest = hashlib.sha256(audio.read_bytes()).hexdigest()
    if truth.get("generator_version") != 2 or truth.get("audio_sha256") != digest:
        raise ValueError("replay requires a hash-matched generator-v2 synthetic fixture")
    # Include primary utterances as Eneo's caller transcript; the omitted
    # backchannels intentionally exercise missing-ASR-content behavior.
    segments = [
        {"start": p["start"], "end": p["end"], "text": p["text"]}
        for p in truth["segments"]
        if p["kind"] == "turn"
    ]
    return digest, segments


def submit(client: httpx.Client, audio: Path, request: JobRequest) -> str:
    fields = request.model_dump(mode="json", exclude_none=True)
    with audio.open("rb") as stream:
        response = client.post(
            "v1/jobs",
            files={"file": (audio.name, stream, "audio/wav")},
            data={k: v if isinstance(v, str) else json.dumps(v) for k, v in fields.items()},
        )
    response.raise_for_status()
    return response.json()["job_id"]


def wait_result(client: httpx.Client, job_id: str, *, timeout=3600) -> TranscriptionResult:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.get(f"v1/jobs/{job_id}")
        status.raise_for_status()
        state = status.json()["status"]
        if state == "completed":
            response = client.get(f"v1/jobs/{job_id}/result")
            response.raise_for_status()
            return TranscriptionResult.model_validate(response.json())
        if state in {"failed", "cancelled"}:
            raise RuntimeError(f"Vemsa job {job_id} {state}; inspect its status for details")
        time.sleep(2)
    raise TimeoutError(f"Vemsa job {job_id} still pending; it can be retrieved later")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", required=True, help="explicit Vemsa service URL, without /v1")
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--task", choices=["diarize", "transcribe"], default="diarize")
    ap.add_argument("--legacy", action="store_true", help="compare without review metadata")
    ap.add_argument("--num-speakers", type=int)
    ap.add_argument("--min-speakers", type=int)
    ap.add_argument("--max-speakers", type=int)
    ap.add_argument("--timeout", type=float, default=3600)
    args = ap.parse_args()
    token = os.environ.get("VEMSA_EVAL_TOKEN")
    if not token:
        ap.error("set VEMSA_EVAL_TOKEN for the chosen test service")
    digest, segments = load_fixture(args.audio, args.truth)
    request = JobRequest(
        task=args.task,
        language="sv",
        include_speaker_review=not args.legacy,
        segments=segments if args.task == "diarize" else None,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )
    with httpx.Client(
        base_url=args.base_url.rstrip("/") + "/",
        headers={"Authorization": f"Bearer {token}"},
        timeout=120,
    ) as client:
        job_id = submit(client, args.audio, request)
        print(f"submitted {job_id}", flush=True)
        result = wait_result(client, job_id, timeout=args.timeout)
    artifact = {
        "harness": "Vemsa HTTP service, actual decoder/alignment/attribution",
        "created_at": datetime.now(UTC).isoformat(),
        "audio_sha256": digest,
        "job_id": job_id,
        "request": request.model_dump(mode="json", exclude_none=True),
        "result": result.model_dump(mode="json"),
    }
    output = args.audio.with_suffix(f".{job_id}.service.json")
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    labels = {s.speaker for s in result.segments if s.speaker is not None}
    print(f"saved {output}; {len(labels)} proposed labels; alignment={result.alignment}")


if __name__ == "__main__":
    main()
