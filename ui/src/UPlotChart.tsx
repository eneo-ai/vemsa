import {useEffect, useRef} from 'react';
import uPlot from 'uplot';

interface Props {
  data: uPlot.AlignedData;
  options: Omit<uPlot.Options, 'width' | 'height'>;
  height?: number;
  /** Rebuild the chart when this changes (series shape, palette). */
  revision: string;
}

/** Thin uPlot host: sizes to its container, rebuilds on `revision`, otherwise only pushes data. */
export function UPlotChart({data, options, height = 200, revision}: Props) {
  const host = useRef<HTMLDivElement>(null);
  const chart = useRef<uPlot | null>(null);
  const latest = useRef({data, options, height});
  latest.current = {data, options, height};

  useEffect(() => {
    const element = host.current;
    if (!element) return;
    const build = () => {
      chart.current?.destroy();
      const {data: initialData, options: currentOptions, height: currentHeight} = latest.current;
      chart.current = new uPlot(
        {...currentOptions, width: element.clientWidth || 600, height: currentHeight},
        initialData,
        element,
      );
    };
    build();
    const observer = new ResizeObserver(() => {
      if (chart.current && element.clientWidth) {
        chart.current.setSize({width: element.clientWidth, height: latest.current.height});
      }
    });
    observer.observe(element);
    return () => {
      observer.disconnect();
      chart.current?.destroy();
      chart.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [revision]);

  useEffect(() => {
    chart.current?.setData(data);
  }, [data]);

  return <div ref={host} className="ops-chart" />;
}
