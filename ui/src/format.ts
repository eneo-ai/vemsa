export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return '–';
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`;
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)} s`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds - minutes * 60);
  if (minutes < 60) return rest ? `${minutes}m ${rest}s` : `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  const restMinutes = minutes - hours * 60;
  if (hours < 48) return restMinutes ? `${hours}h ${restMinutes}m` : `${hours}h`;
  return `${(hours / 24).toFixed(1)} d`;
}

export function formatMinutes(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return '–';
  const minutes = seconds / 60;
  if (minutes >= 1000) return `${(minutes / 60).toFixed(1)} h`;
  if (minutes >= 100) return `${Math.round(minutes)} min`;
  return `${minutes.toFixed(1)} min`;
}

export function formatCount(value: number | null | undefined): string {
  if (value === null || value === undefined) return '–';
  return new Intl.NumberFormat().format(value);
}

export function formatRtf(value: number | null | undefined): string {
  if (value === null || value === undefined) return '–';
  return `${value.toFixed(value < 0.1 ? 3 : 2)}×`;
}

export function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined) return '–';
  return `${Math.round(value)} %`;
}

export function formatBytes(value: number | null | undefined): string {
  if (value === null || value === undefined) return '–';
  const gib = value / 1024 ** 3;
  if (gib >= 100) return `${Math.round(gib)} GiB`;
  if (gib >= 1) return `${gib.toFixed(1)} GiB`;
  return `${Math.round(value / 1024 ** 2)} MiB`;
}

export function formatTime(iso: string | null | undefined): string {
  if (!iso) return '–';
  const date = new Date(iso);
  return new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date);
}

export function formatAgo(iso: string | null | undefined, now: number = Date.now()): string {
  if (!iso) return '–';
  const seconds = Math.max(0, (now - new Date(iso).getTime()) / 1000);
  return `${formatDuration(seconds)} ago`;
}

export function shortId(id: string): string {
  return id.length > 12 ? `${id.slice(0, 8)}…` : id;
}
