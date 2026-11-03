import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'

import { api } from '@/lib/api'
import type { ExperimentRecord, Trace } from '@/lib/types'

export function traced(r: ExperimentRecord): boolean {
  return r.status === 'completed' && r.request.include_trace !== false
}

interface Loaded {
  id: string
  record: ExperimentRecord | null
  trace: Trace | null
  error: string | null
}

/**
 * The experiment selected by `?id=`; without one, the most recent completed
 * experiment that recorded a trace. Loads its record (incl. events) and trace.
 */
export function useExperimentTrace() {
  const [params, setParams] = useSearchParams()
  const [candidates, setCandidates] = useState<ExperimentRecord[] | null>(null)
  const [listError, setListError] = useState<string | null>(null)
  const [loaded, setLoaded] = useState<Loaded | null>(null)

  useEffect(() => {
    api.experiments(100).then((rows) => setCandidates(rows.filter(traced)), (e: Error) => setListError(e.message))
  }, [])

  const selected = params.get('id') ?? candidates?.[0]?.id ?? null

  useEffect(() => {
    if (!selected) return
    let live = true
    Promise.all([api.experiment(selected), api.trace(selected)])
      .then(([record, trace]) => live && setLoaded({ id: selected, record, trace, error: null }))
      .catch((e: Error) => live && setLoaded({ id: selected, record: null, trace: null, error: e.message }))
    return () => { live = false }
  }, [selected])

  // results for a previous selection are never shown for the current one
  const current = loaded && loaded.id === selected ? loaded : null
  return {
    selected,
    candidates,
    record: current?.record ?? null,
    trace: current?.trace ?? null,
    error: listError ?? current?.error ?? null,
    loading: !!selected && !current,
    select: (next: string) => setParams((p) => { p.set('id', next); return p }),
  }
}
