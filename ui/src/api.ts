// Types mirror src/vemsa/ops/router.py responses; timestamps are ISO strings in UTC.
import {useCallback, useEffect, useRef, useState} from 'react';

export type Window = '1h' | '6h' | '24h' | '7d' | '30d';
export const WINDOWS: Window[] = ['1h', '6h', '24h', '7d', '30d'];

export interface ServiceSummary {
  service_version: string;
  environment: string;
  engine: string;
  default_model: string;
  diarization_model: string;
  emissions_model: string;
  worker_concurrency: number;
  gpu_concurrency: number;
  oom_max_attempts: number;
  retention_hours: number;
  stats_retention_days: number;
  max_queued_jobs: number;
  max_queued_jobs_per_client: number;
  min_alignment: string | null;
  diarize_prefer_align: boolean;
  worker_stale_s: number;
  lease_heartbeat_s: number;
  in_process_worker: boolean;
}

export interface Readiness {
  status: 'ready' | 'not_ready';
  service_version: string;
  database_ready: boolean;
  worker_ready: boolean;
  queue_accepting_jobs: boolean;
  queued_jobs: number | null;
}

export interface RunningJob {
  id: string;
  client_id: string;
  task: string;
  stage: string;
  attempt: number;
  started_at: string | null;
  lease_owner: string | null;
  lease_expires_at: string | null;
  elapsed_s: number | null;
}

export interface WorkerSnapshot {
  worker_id: string;
  last_heartbeat: string;
  heartbeat_age_s: number | null;
  alive: boolean;
  sampled_at: string | null;
  hostname: string | null;
  engine: string | null;
  device: string | null;
  gpu_name: string | null;
  in_flight: number | null;
  concurrency: number | null;
  gpu_concurrency: number | null;
  cpu_pct: number | null;
  load1: number | null;
  mem_used_bytes: number | null;
  mem_total_bytes: number | null;
  disk_used_bytes: number | null;
  disk_total_bytes: number | null;
  gpu_util_pct: number | null;
  gpu_mem_used_bytes: number | null;
  gpu_mem_total_bytes: number | null;
  gpu_temp_c: number | null;
}

export interface Overview {
  now: string;
  service: ServiceSummary;
  readiness: Readiness;
  jobs: Record<'queued' | 'running' | 'completed' | 'failed' | 'cancelled', number>;
  oldest_queued_s: number | null;
  running: RunningJob[];
  workers: WorkerSnapshot[];
}

export interface Bucket {
  start: string;
  completed: number;
  failed: number;
  cancelled: number;
  audio_seconds: number;
  processing_s: number;
}

export interface TaskTotals {
  task: string;
  completed: number;
  failed: number;
  cancelled: number;
  audio_seconds: number;
  processing_s: number;
}

export interface Throughput {
  window: Window;
  since: string;
  until: string;
  bucket: string;
  buckets: Bucket[];
  by_task: TaskTotals[];
  totals: Omit<Bucket, 'start'>;
}

export interface TaskPerformance {
  task: string;
  completed: number;
  failed: number;
  cancelled: number;
  retried: number;
  rtf_p50: number | null;
  rtf_p95: number | null;
  queue_wait_p50_s: number | null;
  queue_wait_p95_s: number | null;
  processing_p50_s: number | null;
  processing_p95_s: number | null;
}

export interface StageStat {
  stage: string;
  p50: number;
  p95: number;
  total_s: number;
  jobs: number;
}

export interface Performance {
  window: Window;
  tasks: TaskPerformance[];
  stages: StageStat[];
}

export interface Quality {
  window: Window;
  alignment: {task: string; alignment: string; jobs: number}[];
  errors: {error_class: string; task: string; jobs: number}[];
}

export interface ClientRow {
  client_id: string;
  completed: number;
  failed: number;
  cancelled: number;
  audio_seconds: number;
  processing_s: number;
  rtf_p50: number | null;
}

export interface Clients {
  window: Window;
  clients: ClientRow[];
}

export interface HostPoint {
  t: string;
  cpu_pct: number | null;
  load1: number | null;
  mem_used_bytes: number | null;
  mem_total_bytes: number | null;
  disk_used_bytes: number | null;
  disk_total_bytes: number | null;
  gpu_util_pct: number | null;
  gpu_mem_used_bytes: number | null;
  gpu_mem_total_bytes: number | null;
  gpu_temp_c: number | null;
  in_flight: number | null;
  concurrency: number | null;
}

export interface HostSeries {
  worker_id: string;
  hostname: string | null;
  device: string | null;
  gpu_name: string | null;
  points: HostPoint[];
}

export interface Host {
  window: Window;
  since: string;
  until: string;
  workers: HostSeries[];
}

export interface RecentJob {
  job_id: string;
  client_id: string;
  task: string;
  status: 'completed' | 'failed' | 'cancelled';
  engine: string | null;
  model: string | null;
  language: string | null;
  alignment: string | null;
  device: string | null;
  attempts: number;
  created_at: string;
  started_at: string | null;
  finished_at: string;
  processing_s: number | null;
  audio_seconds: number | null;
  stage_seconds: Record<string, number> | null;
  error_class: string | null;
  queue_wait_s: number | null;
  rtf: number | null;
}

export interface Recent {
  jobs: RecentJob[];
}

const BASE = '/ops/api';
export const REFRESH_MS = 15_000;

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(BASE + path, {headers: {Accept: 'application/json'}});
  if (!response.ok) {
    throw new Error(`${path}: HTTP ${response.status}`);
  }
  return (await response.json()) as T;
}

export interface Loaded<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  updatedAt: number | null;
}

/** Polls one endpoint; keeps the last good payload while refreshing, pauses when the tab is hidden. */
export function useOpsData<T>(path: string): Loaded<T> {
  const [state, setState] = useState<Loaded<T>>({
    data: null,
    error: null,
    loading: true,
    updatedAt: null,
  });
  const inFlight = useRef(false);

  const load = useCallback(async () => {
    if (inFlight.current) return;
    inFlight.current = true;
    setState((previous) => ({...previous, loading: true}));
    try {
      const data = await getJson<T>(path);
      setState({data, error: null, loading: false, updatedAt: Date.now()});
    } catch (error) {
      setState((previous) => ({
        ...previous,
        error: error instanceof Error ? error.message : String(error),
        loading: false,
      }));
    } finally {
      inFlight.current = false;
    }
  }, [path]);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;
    // the first load always runs; only the periodic refresh pauses while hidden
    void load();
    const tick = () => {
      if (cancelled) return;
      if (!document.hidden) void load();
      timer = window.setTimeout(tick, REFRESH_MS);
    };
    const onVisible = () => {
      if (!document.hidden) void load();
    };
    document.addEventListener('visibilitychange', onVisible);
    timer = window.setTimeout(tick, REFRESH_MS);
    return () => {
      cancelled = true;
      if (timer !== undefined) window.clearTimeout(timer);
      document.removeEventListener('visibilitychange', onVisible);
    };
  }, [load]);

  return state;
}
