import { useEffect } from 'react'
import { Route, Routes } from 'react-router-dom'

import { Sidebar } from '@/components/layout/Sidebar'
import { Comparison } from '@/pages/Comparison'
import { ConceptExplorer } from '@/pages/ConceptExplorer'
import { Overview } from '@/pages/Overview'
import { Placeholder } from '@/pages/Placeholder'
import { Playground } from '@/pages/Playground'
import { SettingsPage } from '@/pages/Settings'
import { Timeline } from '@/pages/Timeline'
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
          <Route path="/concepts" element={<ConceptExplorer />} />
          <Route path="/timeline" element={<Timeline />} />
          <Route path="/compare" element={<Comparison />} />
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
