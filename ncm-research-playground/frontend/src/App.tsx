import { useEffect } from 'react'
import { Route, Routes } from 'react-router-dom'

import { Sidebar } from '@/components/layout/Sidebar'
import { Overview } from '@/pages/Overview'
import { Placeholder } from '@/pages/Placeholder'
import { Playground } from '@/pages/Playground'
import { SettingsPage } from '@/pages/Settings'
import { useModels } from '@/store/models'

export default function App() {
  const refresh = useModels((s) => s.refresh)
  const status = useModels((s) => s.status)
  useEffect(() => {
    void refresh()
  }, [refresh])
  // backend may still be starting (checkpoint registry, uvicorn reload): retry
  useEffect(() => {
    if (status !== 'error') return
    const t = setTimeout(() => void refresh(), 3000)
    return () => clearTimeout(t)
  }, [status, refresh])

  return (
    <div className="flex min-h-svh">
      <Sidebar />
      <main className="min-w-0 flex-1">
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/playground" element={<Playground />} />
          <Route path="/concepts" element={<Placeholder title="Concept Explorer" milestone={8}
            description="Inspect the hierarchy of discrete codes NCM generated, level by level."
            detail="Level navigation, code-index heatmaps, codebook distributions and per-level decodes, built from the real traces the backend already records." />} />
          <Route path="/timeline" element={<Placeholder title="Generation Timeline" milestone={8}
            description="Sequential levels, each predicted in one parallel step."
            detail="Per-level execution time, code counts and status from recorded generation events." />} />
          <Route path="/compare" element={<Placeholder title="Model Comparison" milestone={9}
            description="NCM against an autoregressive baseline, side by side."
            detail="Needs the autoregressive baseline runner, which is not implemented yet." />} />
          <Route path="/benchmarks" element={<Placeholder title="Benchmarks" milestone={10}
            description="Reproducible latency, throughput and memory measurements."
            detail="Warm-up runs, repeated measurements, medians and percentiles, CSV export. No benchmark data exists yet." />} />
          <Route path="/history" element={<Placeholder title="Experiment History" milestone={10}
            description="Every recorded run, reopenable and replayable."
            detail="The backend already stores experiments and their events. This view arrives with the benchmarking milestone." />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Routes>
      </main>
    </div>
  )
}
