import {Banner} from '@astryxdesign/core/Banner';
import {HStack} from '@astryxdesign/core/HStack';
import {SegmentedControl, SegmentedControlItem} from '@astryxdesign/core/SegmentedControl';
import {StatusDot} from '@astryxdesign/core/StatusDot';
import {Text} from '@astryxdesign/core/Text';
import {TopNav} from '@astryxdesign/core/TopNav';
import {VStack} from '@astryxdesign/core/VStack';
import {useEffect, useState} from 'react';

import {
  type Clients,
  type Host,
  type Overview,
  type Performance,
  type Quality,
  type Recent,
  type Throughput,
  WINDOWS,
  type Window,
  useOpsData,
} from './api';
import {formatAgo} from './format';
import {HostPanel} from './panels/HostPanel';
import {ClientsPanel, RecentJobsPanel} from './panels/JobsPanel';
import {OverviewPanel} from './panels/OverviewPanel';
import {PerformancePanel, QualityPanel} from './panels/PerformancePanel';
import {ThroughputPanel} from './panels/ThroughputPanel';

const STORAGE_KEY = 'vemsa-ops-window';

function readWindow(): Window {
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (stored && (WINDOWS as string[]).includes(stored)) return stored as Window;
  } catch {
    // storage may be unavailable; the default is fine
  }
  return '24h';
}

export function App() {
  const [range, setRange] = useState<Window>(readWindow);
  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, range);
    } catch {
      // ignore
    }
  }, [range]);

  const overview = useOpsData<Overview>('/overview');
  const throughput = useOpsData<Throughput>(`/throughput?window=${range}`);
  const performance = useOpsData<Performance>(`/performance?window=${range}`);
  const quality = useOpsData<Quality>(`/quality?window=${range}`);
  const clients = useOpsData<Clients>(`/clients?window=${range}`);
  const host = useOpsData<Host>(`/host?window=${range}`);
  const recent = useOpsData<Recent>('/jobs/recent?limit=50');

  const firstError = [overview, throughput, performance, quality, clients, host, recent]
    .map((loaded) => loaded.error)
    .find(Boolean);
  const ready = overview.data?.readiness.status === 'ready';

  return (
    <VStack gap={0} minHeight="100vh">
      <TopNav
        heading="vemsa ops"
        endContent={
          <HStack gap={3} align="center">
            <Text type="supporting" color="secondary">
              {overview.updatedAt ? `updated ${formatAgo(new Date(overview.updatedAt).toISOString())}` : 'loading…'}
            </Text>
            <StatusDot
              variant={overview.data ? (ready ? 'success' : 'error') : 'neutral'}
              label={overview.data ? (ready ? 'service ready' : 'not ready') : 'connecting'}
              isPulsing={ready}
            />
          </HStack>
        }
      />
      <VStack gap={6} padding={4} width="100%">
        <div className="ops-page">
          <VStack gap={6}>
            {firstError ? (
              <Banner status="error" title="Dashboard data unavailable" description={firstError} />
            ) : null}
            <HStack gap={3} align="center" wrap="wrap">
              <SegmentedControl label="Window" value={range} onChange={(value) => setRange(value as Window)} size="sm" layout="hug">
                {WINDOWS.map((option) => (
                  <SegmentedControlItem key={option} value={option} label={option} />
                ))}
              </SegmentedControl>
              <Text type="supporting" color="secondary">
                Throughput, performance, quality, clients and host panels follow this window. Refreshes every 15 s.
              </Text>
            </HStack>
            <Section loaded={overview}>{overview.data ? <OverviewPanel overview={overview.data} /> : null}</Section>
            <Section loaded={throughput}>
              {throughput.data ? <ThroughputPanel throughput={throughput.data} /> : null}
            </Section>
            <Section loaded={performance}>
              {performance.data ? <PerformancePanel performance={performance.data} /> : null}
            </Section>
            <Section loaded={quality}>{quality.data ? <QualityPanel quality={quality.data} /> : null}</Section>
            <Section loaded={host}>{host.data ? <HostPanel host={host.data} /> : null}</Section>
            <Section loaded={clients}>{clients.data ? <ClientsPanel clients={clients.data} /> : null}</Section>
            <Section loaded={recent}>{recent.data ? <RecentJobsPanel recent={recent.data} /> : null}</Section>
          </VStack>
        </div>
      </VStack>
    </VStack>
  );
}

/** Holds the previous render at reduced opacity while a refresh is in flight. */
function Section({loaded, children}: {loaded: {loading: boolean; data: unknown}; children: React.ReactNode}) {
  const stale = loaded.loading && loaded.data !== null;
  return <div className={stale ? 'ops-stale' : undefined}>{children}</div>;
}
