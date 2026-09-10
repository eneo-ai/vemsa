import {Card} from '@astryxdesign/core/Card';
import {Heading} from '@astryxdesign/core/Heading';
import {Text} from '@astryxdesign/core/Text';
import {VStack} from '@astryxdesign/core/VStack';
import type {ReactNode} from 'react';

/** Label above, value in the figure size, optional supporting line below. */
export function StatTile({
  label,
  value,
  detail,
  tone,
}: {
  label: string;
  value: ReactNode;
  detail?: ReactNode;
  tone?: 'default' | 'muted' | 'red' | 'orange' | 'green';
}) {
  return (
    <Card variant={tone ?? 'default'} padding={3} minHeight={96}>
      <VStack gap={1}>
        <Text type="supporting" color="secondary">
          {label}
        </Text>
        <div className="ops-value">{value}</div>
        {detail ? (
          <Text type="supporting" color="secondary">
            {detail}
          </Text>
        ) : null}
      </VStack>
    </Card>
  );
}

export function Panel({
  title,
  description,
  children,
}: {
  title: string;
  description?: string;
  children: ReactNode;
}) {
  return (
    <VStack gap={3} as="section">
      <VStack gap={0.5}>
        <Heading level={2} type="display-3">
          {title}
        </Heading>
        {description ? (
          <Text type="supporting" color="secondary">
            {description}
          </Text>
        ) : null}
      </VStack>
      {children}
    </VStack>
  );
}

export function ChartCard({title, children}: {title: string; children: ReactNode}) {
  return (
    <Card padding={3}>
      <VStack gap={2}>
        <Text type="label" weight="medium">
          {title}
        </Text>
        {children}
      </VStack>
    </Card>
  );
}

export function Empty({children}: {children: ReactNode}) {
  return (
    <Text type="supporting" color="secondary">
      {children}
    </Text>
  );
}

/** Astryx's Table wants index-signature rows; plain interfaces need this bridge. */
export function asRows<T extends object>(items: T[]): (T & Record<string, unknown>)[] {
  return items as (T & Record<string, unknown>)[];
}
