# SentiFolio Dashboard (React)

A React + TypeScript single-page dashboard for the SentiFolio API — a dark,
Bloomberg-terminal-style view of the portfolio signals, risk, forecasts, SHAP
attribution, backtest and the regime finding.

It replaces the earlier Streamlit prototype (in line with supervisor feedback to
adopt an alternative to Streamlit) and consumes the existing FastAPI backend
unchanged.

## Stack

- **Vite** + **React** + **TypeScript**
- **Tailwind CSS** (custom `term-*` terminal palette) — the `@` path alias and
  Tailwind config are set up so shadcn/ui components can be added later
- **TanStack Query** for data fetching / caching / loading & error states
- **Recharts** for the SHAP bar chart and the equity-curve line chart
- **axios** API client

## Run (dev)

The backend must be running first:

```powershell
# from the repo root
.\.venv\Scripts\python.exe -m uvicorn src.serve.api:app --reload   # :8000
```

Then:

```powershell
cd frontend
npm install        # first time only
npm run dev        # http://localhost:5173
```

Vite proxies `/api/*` to `http://127.0.0.1:8000` (see `vite.config.ts`), so there
is no CORS to configure in dev. The backend also allows the Vite origin directly
(CORS middleware in `src/serve/api.py`) for non-proxied use.

## Build

```powershell
npm run build      # type-checks (tsc -b) then bundles to dist/
```

## Layout

```
src/
  main.tsx             # TanStack Query provider
  App.tsx              # top bar (arm toggle) + KPI strip + panel grid
  lib/api.ts           # typed API client for all 8 endpoints
  components/
    Panel.tsx          # terminal panel shell (title bar, loading/error states)
    EquityChart.tsx    # /equity  -> Recharts line chart
    RegimePanel.tsx    # /regime  -> sentiment delta Sharpe by market regime
```

The `arm` toggle (`price_only` / `price+sentiment`) in the top bar is the single
piece of global state; every arm-dependent panel refetches when it changes.
