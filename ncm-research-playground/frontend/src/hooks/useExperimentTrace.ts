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
 * experiment that recorded a trace (and passes `filter`, if given). Loads its
 * record (incl. events) and trace.
 */
export function useExperimentTrace(filter?: (r: ExperimentRecord) => boolean) {
  const [params, setParams] = useSearchParams()
  const [candidates, setCandidates] = useState<ExperimentRecord[] | null>(null)
  const [listError, setListError] = useState<string | null>(null)
  const [loaded, setLoaded] = useState<Loaded | null>(null)

  const [reloads, setReloads] = useState(0)
  useEffect(() => {
    api.experiments(100).then((rows) => setCandidates(rows.filter((r) => traced(r) && (!filter || filter(r)))), (e: Error) => setListError(e.message))
    // filter is expected to be a stable (module-level) function
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reloads])

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
    reload: () => setReloads((n) => n + 1),
  }
}
