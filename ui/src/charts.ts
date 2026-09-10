// Shared uPlot option builders so every chart follows the same mark specs:
// 2px lines, thin rounded bars, hairline grid, crosshair with a full readout.
import uPlot from 'uplot';

import type {ChartPalette} from './theme';

export const toSeconds = (iso: string): number => new Date(iso).getTime() / 1000;

/** Pin the time axis to the queried window so sparse series do not auto-range into years. */
export function windowScale(since: string, until: string): uPlot.Scale {
  return {time: true, range: [toSeconds(since), toSeconds(until)]};
}

export function timeAxis(palette: ChartPalette): uPlot.Axis {
  return {
    stroke: palette.text,
    grid: {stroke: palette.grid, width: 1},
    ticks: {stroke: palette.axis, width: 1},
    font: '11px system-ui, sans-serif',
  };
}

export function valueAxis(palette: ChartPalette, format: (value: number) => string): uPlot.Axis {
  return {
    stroke: palette.text,
    grid: {stroke: palette.grid, width: 1},
    ticks: {show: false},
    font: '11px system-ui, sans-serif',
    size: 56,
    values: (_chart, splits) => splits.map((split) => format(split)),
  };
}

export function baseOptions(palette: ChartPalette): Omit<uPlot.Options, 'width' | 'height' | 'series'> {
  return {
    cursor: {
      x: true,
      y: false,
      points: {size: 8, fill: (chart, index) => String(chart.series[index].stroke ?? palette.text)},
    },
    legend: {show: true, live: true},
    padding: [8, 8, 0, 0],
  };
}

export function lineSeries(
  label: string,
  color: string,
  format: (value: number | null) => string,
): uPlot.Series {
  return {
    label,
    stroke: color,
    width: 2,
    points: {show: false},
    spanGaps: false,
    value: (_chart, value) => format(value as number | null),
  };
}

/** Bars of at most 24px, rounded data-end, with a 2px surface gap between neighbours. */
export function barSeries(
  label: string,
  color: string,
  format: (value: number | null) => string,
): uPlot.Series {
  return {
    label,
    stroke: color,
    fill: color,
    width: 0,
    points: {show: false},
    // anchored to the bucket start: the bar covers the interval it counts
    paths: uPlot.paths.bars!({size: [0.7, 24], gap: 2, radius: 0.15, align: 1}),
    value: (_chart, value) => format(value as number | null),
  };
}
