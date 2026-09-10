# Operator dashboard (`/ops`)

The page served at `/ops` by the api. Vite + React 19, [Astryx](https://astryx.atmeta.com)
components, [uPlot](https://github.com/leeoniya/uPlot) charts. It has no state of its own:
everything on screen is a poll of `/ops/api/*`, which reads PostgreSQL. Operator-facing
behaviour (auth, retention, what is stored) is documented under "Ops dashboard" in
`docs/PRODUCTION.md`; this file is for people changing the page.

## Build and run

```bash
npm ci                     # once, or after package.json changes
npm run build              # typecheck + vite build -> ../src/vemsa/ops/static (gitignored)
npm run dev                # hot reload on :5173, /ops/api proxied to a running api on :8000
npm run typecheck          # tsc --noEmit only
npm run astryx -- component Card   # Astryx CLI: props and usage for any component
```

`npm run build` writes into the Python package so `uv run start` and the container image
serve the result without a separate host. The Dockerfile builds it in a `node:24-slim`
stage, CI builds it on every push, and the devcontainer builds it in `post-create.sh`. A
checkout without a build still serves `/ops`: the api answers with a short "UI not built"
page while the JSON endpoints keep working.

The Astryx components are shipped pre-compiled, so no StyleX plugin is configured. If you
add custom `stylex.create` styles or swizzle a component, add `@stylexjs/rollup-plugin` to
`vite.config.ts` first.

## Layout of `src/`

| File | Role |
| --- | --- |
| `main.tsx` | Mounts `<App>` inside Astryx's `<Theme>` (neutral theme, `mode="system"` so dark mode follows the OS) and imports the reset, core, theme, and uPlot stylesheets. |
| `App.tsx` | Top nav, the window picker, one `useOpsData` hook per endpoint, and the panel order. The chosen window is remembered in `localStorage`. |
| `api.ts` | TypeScript types mirroring every response of `src/vemsa/ops/router.py`, plus `useOpsData`, the polling hook. |
| `format.ts` | Number and time formatting (durations, audio minutes, bytes, real-time factor, local timestamps). |
| `theme.ts` | Chart palette for light and dark, and the `usePalette` hook keyed on `prefers-color-scheme`. |
| `charts.ts` | uPlot option builders: time and value axes, line and bar series, `windowScale`. |
| `UPlotChart.tsx` | The one uPlot host component: sizes to its container, rebuilds on `revision`, otherwise only pushes data. |
| `components.tsx` | `StatTile`, `Panel`, `ChartCard`, `Empty`, and `asRows` (the index-signature bridge Astryx's `Table` needs). |
| `panels/*.tsx` | One file per section: overview, throughput, performance + quality, host, jobs + clients. |
| `styles.css` | Page background, the `.ops-*` helpers, and the uPlot legend styling. |

## How data flows

`useOpsData(path)` fetches `/ops/api` + `path` on mount, then every 15 s while the tab is
visible (a hidden tab pauses the refresh, never the first load). It keeps the last good
payload while a refresh is in flight; `App` wraps each panel in a `Section` that drops
opacity during that refresh so the layout never jumps. Errors surface in one banner at the
top; the panels below keep their last data.

The window picker (1h, 6h, 24h, 7d, 30d) is a query parameter on every windowed endpoint.
Bucket sizes are chosen server-side (see `WINDOWS` in `src/vemsa/ops/queries.py`) and come
back in the payload as `since`, `until`, and `bucket`; the charts pin their x-axis to
`since`..`until` with `windowScale` so a worker with two samples does not auto-range into
years. Timestamps arrive as UTC ISO strings and are rendered in the browser's local zone.

## Adding a metric

1. Add the query to `src/vemsa/ops/queries.py` and expose it from `router.py`; extend
   `tests/test_ops_api.py` with seeded rows.
2. Mirror the response shape in `api.ts`.
3. Render it in the matching `panels/*.tsx`, or add a panel and list it in `App.tsx`.
4. Rebuild (`npm run build`) and open `/ops` against `uv run start`.

Never return `request_json`, `result_json`, `audio_path`, or `error` from an ops endpoint:
they carry source URLs, transcripts, and internal messages. `tests/test_ops_api.py`
asserts that no ops body contains them or any configured secret.

## Chart conventions

The charts follow one small rulebook so they read as one system:

- One y-axis per chart. Two measures of different scale are two charts (audio minutes and
  job counts are separate for this reason).
- Series colours are the validated categorical slots in `theme.ts`, assigned in a fixed
  order per chart (completed, failed, cancelled; host CPU, GPU). Status meaning is carried
  by the legend label, never by colour alone.
- Lines are 2 px, bars are at most 24 px with a 2 px surface gap and are anchored to the
  start of the bucket they count. Series with fewer than three points draw markers instead
  of a line.
- The uPlot legend doubles as the hover readout: it lists every series at the cursor's x.
- Text never wears a series colour; values use the Astryx text tokens.

## Astryx notes

- Import components from their subpath (`@astryxdesign/core/Card`), not the package root,
  so the bundle only carries what is used.
- `Table` in data-driven mode wants rows that extend `Record<string, unknown>`; wrap plain
  interface arrays with `asRows` from `components.tsx`.
- Component props and examples: `npm run astryx -- component <Name>`; theme and token
  reference: `npm run astryx -- docs theme` and `npm run astryx -- docs tokens`.
- Pin exact versions in `package.json`; Astryx is pre-1.0 and the component surface used
  here is deliberately small so an upgrade stays cheap.
