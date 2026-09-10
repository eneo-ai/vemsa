import {Grid} from '@astryxdesign/core/Grid';
import {Table} from '@astryxdesign/core/Table';
import {useMemo} from 'react';
import type uPlot from 'uplot';

import type {Throughput} from '../api';
import {barSeries, baseOptions, timeAxis, toSeconds, valueAxis, windowScale} from '../charts';
import {ChartCard, Empty, Panel, StatTile, asRows} from '../components';
import {formatCount, formatMinutes, formatRtf} from '../format';
import {usePalette} from '../theme';
import {UPlotChart} from '../UPlotChart';

const minutes = (value: number | null) => (value === null ? '–' : (value / 60).toFixed(1));
const count = (value: number | null) => (value === null ? '–' : String(value));

export function ThroughputPanel({throughput}: {throughput: Throughput}) {
  const palette = usePalette();
  const {buckets, totals, by_task: byTask, window, since, until} = throughput;

  const audioData = useMemo<uPlot.AlignedData>(
    () => [buckets.map((b) => toSeconds(b.start)), buckets.map((b) => b.audio_seconds)],
    [buckets],
  );
  // stacked: draw cumulative totals back to front so each layer shows its own share
  const jobsData = useMemo<uPlot.AlignedData>(() => {
    const completed = buckets.map((b) => b.completed);
    const failed = buckets.map((b, i) => completed[i] + b.failed);
    const cancelled = buckets.map((b, i) => failed[i] + b.cancelled);
    return [buckets.map((b) => toSeconds(b.start)), cancelled, failed, completed];
  }, [buckets]);

  const audioOptions = useMemo(
    () => ({
      ...baseOptions(palette),
      series: [
        {},
        barSeries('Audio minutes', palette.series[0], (v) => minutes(v as number | null)),
      ],
      axes: [timeAxis(palette), valueAxis(palette, (v) => (v / 60).toFixed(0))],
      scales: {x: windowScale(since, until), y: {range: [0, null] as [number, number | null]}},
    }),
    [palette, since, until],
  );
  const jobsOptions = useMemo(
    () => ({
      ...baseOptions(palette),
      series: [
        {},
        {
          ...barSeries('Cancelled', palette.series[2], count),
          value: (_u: uPlot, _v: number | null, _s: number, idx: number | null) =>
            idx === null ? '–' : String(buckets[idx]?.cancelled ?? 0),
        },
        {
          ...barSeries('Failed', palette.series[1], count),
          value: (_u: uPlot, _v: number | null, _s: number, idx: number | null) =>
            idx === null ? '–' : String(buckets[idx]?.failed ?? 0),
        },
        {
          ...barSeries('Completed', palette.series[0], count),
          value: (_u: uPlot, _v: number | null, _s: number, idx: number | null) =>
            idx === null ? '–' : String(buckets[idx]?.completed ?? 0),
        },
      ],
      axes: [timeAxis(palette), valueAxis(palette, (v) => String(v))],
      scales: {x: windowScale(since, until), y: {range: [0, null] as [number, number | null]}},
    }),
    [palette, buckets, since, until],
  );

  const finished = totals.completed + totals.failed + totals.cancelled;
  const rtf = totals.audio_seconds > 0 ? totals.processing_s / totals.audio_seconds : null;
  return (
    <Panel
      title="Throughput"
      description={`Finished jobs in the last ${window}, by the ${throughput.bucket.replace('PT', '').toLowerCase()} bucket they finished in (UTC-aligned buckets, shown in your local time).`}
    >
      <Grid columns={{minWidth: 170}} gap={2}>
        <StatTile label="Audio processed" value={formatMinutes(totals.audio_seconds)} detail="completed jobs" />
        <StatTile label="Jobs finished" value={formatCount(finished)} detail={`${formatCount(totals.completed)} completed`} />
        <StatTile
          label="Failed"
          value={formatCount(totals.failed)}
          detail={finished ? `${((100 * totals.failed) / finished).toFixed(1)} % of finished` : undefined}
          tone={totals.failed > 0 && totals.failed / Math.max(1, finished) > 0.05 ? 'orange' : 'default'}
        />
        <StatTile label="Cancelled" value={formatCount(totals.cancelled)} />
        <StatTile
          label="Overall real-time factor"
          value={formatRtf(rtf)}
          detail="processing seconds per audio second"
        />
      </Grid>
      {buckets.length === 0 ? (
        <Empty>Nothing finished in this window.</Empty>
      ) : (
        <Grid columns={{minWidth: 420}} gap={2}>
          <ChartCard title="Audio minutes processed">
            <UPlotChart data={audioData} options={audioOptions} revision={`audio-${palette.surface}-${since}`} />
          </ChartCard>
          <ChartCard title="Jobs finished by outcome">
            <UPlotChart data={jobsData} options={jobsOptions} revision={`jobs-${palette.surface}-${since}`} />
          </ChartCard>
        </Grid>
      )}
      {byTask.length > 0 ? (
        <ChartCard title="By task">
          <Table
            data={asRows(byTask)}
            idKey="task"
            density="compact"
            columns={[
              {key: 'task', header: 'Task'},
              {key: 'completed', header: 'Completed', align: 'end'},
              {key: 'failed', header: 'Failed', align: 'end'},
              {key: 'cancelled', header: 'Cancelled', align: 'end'},
              {
                key: 'audio_seconds',
                header: 'Audio',
                align: 'end',
                renderCell: (row) => <span className="ops-num">{formatMinutes(row.audio_seconds)}</span>,
              },
              {
                key: 'processing_s',
                header: 'Real-time factor',
                align: 'end',
                renderCell: (row) => (
                  <span className="ops-num">
                    {formatRtf(row.audio_seconds > 0 ? row.processing_s / row.audio_seconds : null)}
                  </span>
                ),
              },
            ]}
          />
        </ChartCard>
      ) : null}
    </Panel>
  );
}
