import {Grid} from '@astryxdesign/core/Grid';
import {useMemo} from 'react';
import type uPlot from 'uplot';

import type {Host, HostSeries} from '../api';
import {baseOptions, lineSeries, timeAxis, toSeconds, valueAxis, windowScale} from '../charts';
import {ChartCard, Empty, Panel} from '../components';
import {shortId} from '../format';
import {type ChartPalette, usePalette, wash} from '../theme';
import {UPlotChart} from '../UPlotChart';

const GIB = 1024 ** 3;
const pct = (value: number | null) => (value === null ? '–' : `${value.toFixed(0)} %`);
const gib = (value: number | null) => (value === null ? '–' : `${value.toFixed(1)} GiB`);
const num = (value: number | null) => (value === null ? '–' : value.toFixed(2));

export function HostPanel({host}: {host: Host}) {
  const palette = usePalette();
  return (
    <Panel
      title="Host and GPU"
      description="Sampled by each worker on its heartbeat. CPU, memory and disk are host-wide readings, not the container's own share."
    >
      {host.workers.length === 0 ? (
        <Empty>No samples in this window.</Empty>
      ) : (
        host.workers.map((worker) => (
          <WorkerCharts
            key={worker.worker_id}
            worker={worker}
            palette={palette}
            since={host.since}
            until={host.until}
          />
        ))
      )}
    </Panel>
  );
}

function WorkerCharts({
  worker,
  palette,
  since,
  until,
}: {
  worker: HostSeries;
  palette: ChartPalette;
  since: string;
  until: string;
}) {
  const hasGpu = worker.device === 'cuda';
  // a line needs two points; a lone sample is drawn as a marker instead
  const sparse = worker.points.length < 3;
  const line = (label: string, color: string, format: (v: number | null) => string) => ({
    ...lineSeries(label, color, format),
    points: {show: sparse, size: 8},
  });
  const xs = useMemo(() => worker.points.map((p) => toSeconds(p.t)), [worker.points]);

  const utilisation = useMemo<uPlot.AlignedData>(
    () => [
      xs,
      worker.points.map((p) => p.cpu_pct),
      ...(hasGpu ? [worker.points.map((p) => p.gpu_util_pct)] : []),
    ],
    [xs, worker.points, hasGpu],
  );
  const memory = useMemo<uPlot.AlignedData>(
    () => [
      xs,
      worker.points.map((p) => (p.mem_used_bytes === null ? null : p.mem_used_bytes / GIB)),
      ...(hasGpu
        ? [worker.points.map((p) => (p.gpu_mem_used_bytes === null ? null : p.gpu_mem_used_bytes / GIB))]
        : []),
    ],
    [xs, worker.points, hasGpu],
  );
  const load = useMemo<uPlot.AlignedData>(
    () => [xs, worker.points.map((p) => p.load1), worker.points.map((p) => p.in_flight)],
    [xs, worker.points],
  );

  const utilisationOptions = useMemo(
    () => ({
      ...baseOptions(palette),
      series: [
        {},
        {...line('Host CPU', palette.series[0], pct), fill: wash(palette.series[0])},
        ...(hasGpu ? [{...line('GPU', palette.series[1], pct), fill: wash(palette.series[1])}] : []),
      ],
      axes: [timeAxis(palette), valueAxis(palette, (v) => `${v} %`)],
      scales: {x: windowScale(since, until), y: {range: [0, 100] as [number, number]}},
    }),
    [palette, hasGpu, sparse, since, until],
  );
  const memoryMax = Math.max(
    ...worker.points.map((p) => p.mem_total_bytes ?? 0),
    ...worker.points.map((p) => p.gpu_mem_total_bytes ?? 0),
  );
  const memoryOptions = useMemo(
    () => ({
      ...baseOptions(palette),
      series: [
        {},
        line('Host memory used', palette.series[0], gib),
        ...(hasGpu ? [line('GPU memory used', palette.series[1], gib)] : []),
      ],
      axes: [timeAxis(palette), valueAxis(palette, (v) => `${v} GiB`)],
      scales: {
        x: windowScale(since, until),
        y: {range: [0, memoryMax > 0 ? memoryMax / GIB : null] as [number, number | null]},
      },
    }),
    [palette, hasGpu, memoryMax, sparse, since, until],
  );
  const loadOptions = useMemo(
    () => ({
      ...baseOptions(palette),
      series: [
        {},
        line('Load (1 min)', palette.series[0], num),
        line('Jobs in flight', palette.series[2], num),
      ],
      axes: [timeAxis(palette), valueAxis(palette, (v) => String(v))],
      scales: {x: windowScale(since, until), y: {range: [0, null] as [number, number | null]}},
    }),
    [palette, sparse, since, until],
  );

  const title = `${worker.hostname ?? 'worker'} · ${shortId(worker.worker_id)}${
    worker.gpu_name ? ` · ${worker.gpu_name}` : hasGpu ? ' · GPU' : ' · CPU only'
  }`;
  const revision = `${palette.surface}-${hasGpu}-${sparse}-${since}`;
  return (
    <ChartCard title={title}>
      <Grid columns={{minWidth: 360}} gap={2}>
        <div>
          <UPlotChart data={utilisation} options={utilisationOptions} height={180} revision={`u-${revision}`} />
        </div>
        <div>
          <UPlotChart data={memory} options={memoryOptions} height={180} revision={`m-${revision}-${memoryMax}`} />
        </div>
        <div>
          <UPlotChart data={load} options={loadOptions} height={180} revision={`l-${revision}`} />
        </div>
      </Grid>
    </ChartCard>
  );
}
