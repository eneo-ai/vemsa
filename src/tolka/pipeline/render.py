from tolka.jobs.models import Segment


def _hms(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def render_text(segments: list[Segment]) -> str:
    """Render segments as one speaker-labelled, timestamped line each."""
    lines = []
    for segment in segments:
        prefix = f"[{_hms(segment.start)} - {_hms(segment.end)}]"
        if segment.speaker:
            prefix += f" {segment.speaker}:"
        lines.append(f"{prefix} {segment.text}")
    return "\n".join(lines)
