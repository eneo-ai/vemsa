import {Grid} from '@astryxdesign/core/Grid';
import {Table} from '@astryxdesign/core/Table';

import type {Performance, Quality} from '../api';
import {ChartCard, Empty, Panel, StatTile, asRows} from '../components';
import {formatCount, formatDuration, formatRtf} from '../format';

export function PerformancePanel({performance}: {performance: Performance}) {
  const {tasks, stages} = performance;
  return (
    <Panel
      title="Performance"
      description="Medians and 95th percentiles over finished jobs in the window. Real-time factor is processing seconds per audio second: below 1 means faster than real time."
    >
      {tasks.length === 0 ? (
        <Empty>No finished jobs in this window.</Empty>
      ) : (
        tasks.map((task) => (
          <ChartCard key={task.task} title={`task = ${task.task}`}>
            <Grid columns={{minWidth: 170}} gap={2}>
              <StatTile
                label="Real-time factor"
                value={formatRtf(task.rtf_p50)}
                detail={`p95 ${formatRtf(task.rtf_p95)}`}
                tone={task.rtf_p95 !== null && task.rtf_p95 > 1 ? 'orange' : 'default'}
              />
              <StatTile
                label="Queue wait"
                value={formatDuration(task.queue_wait_p50_s)}
                detail={`p95 ${formatDuration(task.queue_wait_p95_s)}`}
              />
              <StatTile
                label="Processing time"
                value={formatDuration(task.processing_p50_s)}
                detail={`p95 ${formatDuration(task.processing_p95_s)}`}
              />
              <StatTile
                label="Finished"
                value={formatCount(task.completed + task.failed + task.cancelled)}
                detail={`${formatCount(task.completed)} completed · ${formatCount(task.failed)} failed`}
              />
              <StatTile
                label="Retried jobs"
                value={formatCount(task.retried)}
                detail="needed more than one claim (out-of-memory or lost lease)"
                tone={task.retried > 0 ? 'orange' : 'default'}
              />
            </Grid>
          </ChartCard>
        ))
      )}
      {stages.length > 0 ? (
        <ChartCard title="Time per pipeline stage (completed jobs)">
          <Table
            data={asRows(stages)}
            idKey="stage"
            density="compact"
            columns={[
              {key: 'stage', header: 'Stage'},
              {key: 'jobs', header: 'Jobs', align: 'end'},
              {
                key: 'p50',
                header: 'Median',
                align: 'end',
                renderCell: (row) => <span className="ops-num">{formatDuration(row.p50)}</span>,
              },
              {
                key: 'p95',
                header: 'p95',
                align: 'end',
                renderCell: (row) => <span className="ops-num">{formatDuration(row.p95)}</span>,
              },
              {
                key: 'total_s',
                header: 'Total',
                align: 'end',
                renderCell: (row) => <span className="ops-num">{formatDuration(row.total_s)}</span>,
              },
            ]}
          />
        </ChartCard>
      ) : null}
    </Panel>
  );
}

export function QualityPanel({quality}: {quality: Quality}) {
  const alignmentRows = quality.alignment.map((row) => ({
    ...row,
    id: `${row.task}/${row.alignment}`,
  }));
  const errorRows = quality.errors.map((row) => ({...row, id: `${row.task}/${row.error_class}`}));
  return (
    <Panel
      title="Quality"
      description="Word-timestamp rung of completed jobs (the quality tiers only ever produce `forced`) and failure classes."
    >
      <Grid columns={{minWidth: 360}} gap={2}>
        <ChartCard title="Alignment rung">
          {alignmentRows.length === 0 ? (
            <Empty>No completed jobs in this window.</Empty>
          ) : (
            <Table
              data={asRows(alignmentRows)}
              idKey="id"
              density="compact"
              columns={[
                {key: 'task', header: 'Task'},
                {key: 'alignment', header: 'Alignment'},
                {key: 'jobs', header: 'Jobs', align: 'end'},
              ]}
            />
          )}
        </ChartCard>
        <ChartCard title="Failures by class">
          {errorRows.length === 0 ? (
            <Empty>No failures in this window.</Empty>
          ) : (
            <Table
              data={asRows(errorRows)}
              idKey="id"
              density="compact"
              columns={[
                {key: 'error_class', header: 'Exception'},
                {key: 'task', header: 'Task'},
                {key: 'jobs', header: 'Jobs', align: 'end'},
              ]}
            />
          )}
        </ChartCard>
      </Grid>
    </Panel>
  );
}
