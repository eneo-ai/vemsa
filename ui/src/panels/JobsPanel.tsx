import {Badge} from '@astryxdesign/core/Badge';
import {Grid} from '@astryxdesign/core/Grid';
import {Table} from '@astryxdesign/core/Table';

import type {Clients, Recent, RecentJob} from '../api';
import {ChartCard, Empty, Panel, asRows} from '../components';
import {formatCount, formatDuration, formatMinutes, formatRtf, formatTime, shortId} from '../format';

const STATUS_VARIANT: Record<RecentJob['status'], 'success' | 'error' | 'neutral'> = {
  completed: 'success',
  failed: 'error',
  cancelled: 'neutral',
};

export function RecentJobsPanel({recent}: {recent: Recent}) {
  return (
    <Panel title="Recent jobs" description="The latest finished jobs, newest first.">
      {recent.jobs.length === 0 ? (
        <Empty>No job has finished yet.</Empty>
      ) : (
        <Table
          data={asRows(recent.jobs)}
          idKey="job_id"
          density="compact"
          textOverflow="truncate"
          hasHover
          columns={[
            {
              key: 'finished_at',
              header: 'Finished',
              renderCell: (job) => <span className="ops-num">{formatTime(job.finished_at)}</span>,
            },
            {
              key: 'job_id',
              header: 'Job',
              renderCell: (job) => <span className="ops-mono">{shortId(job.job_id)}</span>,
            },
            {key: 'client_id', header: 'Client'},
            {key: 'task', header: 'Task'},
            {
              key: 'status',
              header: 'Status',
              renderCell: (job) => (
                <Badge
                  variant={STATUS_VARIANT[job.status]}
                  label={job.status === 'failed' && job.error_class ? job.error_class : job.status}
                />
              ),
            },
            {
              key: 'audio_seconds',
              header: 'Audio',
              align: 'end',
              renderCell: (job) => <span className="ops-num">{formatMinutes(job.audio_seconds)}</span>,
            },
            {
              key: 'queue_wait_s',
              header: 'Wait',
              align: 'end',
              renderCell: (job) => <span className="ops-num">{formatDuration(job.queue_wait_s)}</span>,
            },
            {
              key: 'processing_s',
              header: 'Processing',
              align: 'end',
              renderCell: (job) => <span className="ops-num">{formatDuration(job.processing_s)}</span>,
            },
            {
              key: 'rtf',
              header: 'RTF',
              align: 'end',
              renderCell: (job) => <span className="ops-num">{formatRtf(job.rtf)}</span>,
            },
            {key: 'alignment', header: 'Alignment', renderCell: (job) => job.alignment ?? '–'},
            {key: 'attempts', header: 'Attempts', align: 'end'},
            {key: 'device', header: 'Device', renderCell: (job) => job.device ?? '–'},
          ]}
        />
      )}
    </Panel>
  );
}

export function ClientsPanel({clients}: {clients: Clients}) {
  return (
    <Panel title="Clients" description="Consumers by audio processed in the window.">
      {clients.clients.length === 0 ? (
        <Empty>No finished jobs in this window.</Empty>
      ) : (
        <Grid columns={1}>
          <ChartCard title="Per client">
            <Table
              data={asRows(clients.clients)}
              idKey="client_id"
              density="compact"
              columns={[
                {key: 'client_id', header: 'Client'},
                {
                  key: 'audio_seconds',
                  header: 'Audio',
                  align: 'end',
                  renderCell: (row) => <span className="ops-num">{formatMinutes(row.audio_seconds)}</span>,
                },
                {
                  key: 'completed',
                  header: 'Completed',
                  align: 'end',
                  renderCell: (row) => <span className="ops-num">{formatCount(row.completed)}</span>,
                },
                {
                  key: 'failed',
                  header: 'Failed',
                  align: 'end',
                  renderCell: (row) => <span className="ops-num">{formatCount(row.failed)}</span>,
                },
                {
                  key: 'cancelled',
                  header: 'Cancelled',
                  align: 'end',
                  renderCell: (row) => <span className="ops-num">{formatCount(row.cancelled)}</span>,
                },
                {
                  key: 'rtf_p50',
                  header: 'Median RTF',
                  align: 'end',
                  renderCell: (row) => <span className="ops-num">{formatRtf(row.rtf_p50)}</span>,
                },
              ]}
            />
          </ChartCard>
        </Grid>
      )}
    </Panel>
  );
}
