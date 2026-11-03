import { useEffect } from 'react'
import { Route, Routes } from 'react-router-dom'

import { Sidebar } from '@/components/layout/Sidebar'
import { Benchmarks } from '@/pages/Benchmarks'
import { Comparison } from '@/pages/Comparison'
import { ConceptExplorer } from '@/pages/ConceptExplorer'
import { History } from '@/pages/History'
import { Overview } from '@/pages/Overview'
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
          <Route path="/benchmarks" element={<Benchmarks />} />
          <Route path="/history" element={<History />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Routes>
      </main>
    </div>
  )
}
