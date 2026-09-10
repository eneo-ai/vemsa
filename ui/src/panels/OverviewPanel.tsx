import {Badge} from '@astryxdesign/core/Badge';
import {Grid} from '@astryxdesign/core/Grid';
import {HStack} from '@astryxdesign/core/HStack';
import {ProgressBar} from '@astryxdesign/core/ProgressBar';
import {StatusDot} from '@astryxdesign/core/StatusDot';
import {Table} from '@astryxdesign/core/Table';
import {Text} from '@astryxdesign/core/Text';
import {VStack} from '@astryxdesign/core/VStack';

import type {Overview, RunningJob, WorkerSnapshot} from '../api';
import {ChartCard, Empty, Panel, StatTile, asRows} from '../components';
import {
  formatAgo,
  formatBytes,
  formatCount,
  formatDuration,
  formatPercent,
  shortId,
} from '../format';

export function OverviewPanel({overview}: {overview: Overview}) {
  const {jobs, readiness, service, workers, running} = overview;
  const alive = workers.filter((worker) => worker.alive);
  const capacity = alive.reduce((sum, worker) => sum + (worker.concurrency ?? 0), 0);
  const inFlight = alive.reduce((sum, worker) => sum + (worker.in_flight ?? 0), 0);
  const oldest = overview.oldest_queued_s;
  const queueTone = oldest !== null && oldest > 600 ? 'orange' : 'default';
  return (
    <Panel title="Right now">
      <Grid columns={{minWidth: 170}} gap={2}>
        <StatTile
          label="Admission"
          value={
            <Badge
              variant={readiness.queue_accepting_jobs ? 'success' : 'error'}
              label={readiness.queue_accepting_jobs ? 'Accepting jobs' : 'Rejecting'}
            />
          }
          detail={
            readiness.database_ready
              ? readiness.worker_ready
                ? 'Database and worker ready'
                : 'No recent worker heartbeat'
              : 'Database unreachable'
          }
        />
        <StatTile
          label="Queued"
          value={formatCount(jobs.queued)}
          detail={`cap ${formatCount(service.max_queued_jobs)} active`}
          tone={jobs.queued >= service.max_queued_jobs ? 'red' : 'default'}
        />
        <StatTile
          label="Oldest queued"
          value={oldest === null ? '–' : formatDuration(oldest)}
          detail="time since the head of the queue was submitted"
          tone={queueTone}
        />
        <StatTile
          label="Running"
          value={formatCount(jobs.running)}
          detail={`${inFlight} of ${capacity} slots in use across ${alive.length} live worker${
            alive.length === 1 ? '' : 's'
          }`}
        />
        <StatTile
          label="Retained results"
          value={formatCount(jobs.completed + jobs.failed + jobs.cancelled)}
          detail={`${formatCount(jobs.failed)} failed · ${formatCount(jobs.cancelled)} cancelled · purged after ${service.retention_hours} h`}
        />
        <StatTile
          label="Engine"
          value={service.engine}
          detail={`${service.worker_concurrency} job${service.worker_concurrency === 1 ? '' : 's'} / ${service.gpu_concurrency} GPU slot${service.gpu_concurrency === 1 ? '' : 's'} per worker · v${service.service_version}`}
        />
      </Grid>
      <Grid columns={{minWidth: 360}} gap={2}>
        {workers.length === 0 ? (
          <Empty>No worker has reported a heartbeat in the last day.</Empty>
        ) : (
          workers.map((worker) => <WorkerCard key={worker.worker_id} worker={worker} />)
        )}
      </Grid>
      <RunningJobs jobs={running} />
    </Panel>
  );
}

function WorkerCard({worker}: {worker: WorkerSnapshot}) {
  const memPct =
    worker.mem_used_bytes !== null && worker.mem_total_bytes
      ? (100 * worker.mem_used_bytes) / worker.mem_total_bytes
      : null;
  const diskPct =
    worker.disk_used_bytes !== null && worker.disk_total_bytes
      ? (100 * worker.disk_used_bytes) / worker.disk_total_bytes
      : null;
  const gpuMemPct =
    worker.gpu_mem_used_bytes !== null && worker.gpu_mem_total_bytes
      ? (100 * worker.gpu_mem_used_bytes) / worker.gpu_mem_total_bytes
      : null;
  const title = `${worker.hostname ?? 'worker'} · ${shortId(worker.worker_id)}`;
  return (
    <ChartCard title={title}>
      <HStack gap={2} align="center" wrap="wrap">
        <StatusDot
          variant={worker.alive ? 'success' : 'error'}
          label={worker.alive ? 'live' : 'silent'}
          isPulsing={worker.alive}
        />
        <Text type="supporting" color="secondary">
          heartbeat {formatAgo(worker.last_heartbeat)}
        </Text>
        {worker.device ? (
          <Badge
            variant={worker.device === 'cuda' ? 'info' : 'neutral'}
            label={worker.device === 'cuda' ? (worker.gpu_name ?? 'GPU') : 'CPU only'}
          />
        ) : null}
        {worker.engine ? <Badge variant="neutral" label={worker.engine} /> : null}
      </HStack>
      <VStack gap={1.5}>
        <ProgressBar
          label={`Jobs in flight · ${worker.in_flight ?? 0} of ${worker.concurrency ?? '?'}`}
          value={worker.in_flight ?? 0}
          max={Math.max(1, worker.concurrency ?? 1)}
          variant="accent"
        />
        <ProgressBar
          label={`Host CPU · ${formatPercent(worker.cpu_pct)} · load ${worker.load1?.toFixed(1) ?? '–'}`}
          value={worker.cpu_pct ?? 0}
          variant={severity(worker.cpu_pct)}
        />
        <ProgressBar
          label={`Host memory · ${formatBytes(worker.mem_used_bytes)} of ${formatBytes(worker.mem_total_bytes)}`}
          value={memPct ?? 0}
          variant={severity(memPct)}
        />
        <ProgressBar
          label={`Work dir disk · ${formatBytes(worker.disk_used_bytes)} of ${formatBytes(worker.disk_total_bytes)}`}
          value={diskPct ?? 0}
          variant={severity(diskPct)}
        />
        {worker.device === 'cuda' ? (
          <>
            <ProgressBar
              label={`GPU utilisation · ${formatPercent(worker.gpu_util_pct)}${
                worker.gpu_temp_c !== null ? ` · ${Math.round(worker.gpu_temp_c)} °C` : ''
              }`}
              value={worker.gpu_util_pct ?? 0}
              variant="accent"
            />
            <ProgressBar
              label={`GPU memory · ${formatBytes(worker.gpu_mem_used_bytes)} of ${formatBytes(worker.gpu_mem_total_bytes)}`}
              value={gpuMemPct ?? 0}
              variant={severity(gpuMemPct)}
            />
          </>
        ) : null}
      </VStack>
    </ChartCard>
  );
}

function severity(pct: number | null): 'accent' | 'warning' | 'error' {
  if (pct === null) return 'accent';
  if (pct >= 95) return 'error';
  if (pct >= 80) return 'warning';
  return 'accent';
}

function RunningJobs({jobs}: {jobs: RunningJob[]}) {
  if (jobs.length === 0) {
    return <Empty>No job is running.</Empty>;
  }
  return (
    <ChartCard title={`Running jobs · ${jobs.length}`}>
      <Table
        data={asRows(jobs)}
        idKey="id"
        density="compact"
        textOverflow="truncate"
        columns={[
          {key: 'id', header: 'Job', renderCell: (job) => <span className="ops-mono">{shortId(job.id)}</span>},
          {key: 'client_id', header: 'Client'},
          {key: 'task', header: 'Task'},
          {key: 'stage', header: 'Stage', renderCell: (job) => <Badge variant="info" label={job.stage} />},
          {key: 'attempt', header: 'Attempt', align: 'end'},
          {
            key: 'elapsed_s',
            header: 'Elapsed',
            align: 'end',
            renderCell: (job) => <span className="ops-num">{formatDuration(job.elapsed_s)}</span>,
          },
          {
            key: 'lease_owner',
            header: 'Worker',
            renderCell: (job) => (
              <span className="ops-mono">{job.lease_owner ? shortId(job.lease_owner) : '–'}</span>
            ),
          },
        ]}
      />
    </ChartCard>
  );
}
