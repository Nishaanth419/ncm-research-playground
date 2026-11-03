import { BarChart3, FlaskConical, Inbox, RefreshCw } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'

import { EmptyState, PageHeader, Panel, StatTile, StatusDot } from '@/components/common'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { api } from '@/lib/api'
import { fmtParams, fmtRelative, fmtSeconds } from '@/lib/format'
import type { ExperimentRecord } from '@/lib/types'
import { useModels } from '@/store/models'

const LIMIT = 500

function useExperiments() {
  const [rows, setRows] = useState<ExperimentRecord[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const load = () => api.experiments(LIMIT).then(setRows, (e: Error) => setError(e.message))
  useEffect(() => {
    void load()
  }, [])
  return { rows, error, reload: load }
}

export function Overview() {
  const { models, status, refresh, device } = useModels()
  const { rows, error, reload } = useExperiments()

  const latency = useMemo(() => {
    const by = new Map<string, { n: number; total: number; perSample: number }>()
    for (const r of rows ?? []) {
      if (r.status !== 'completed' || !r.timing) continue
      const k = by.get(r.model) ?? { n: 0, total: 0, perSample: 0 }
      k.n += 1
      k.total += r.timing.total_s
      k.perSample += r.timing.total_s / Math.max(1, r.request.num_samples)
      by.set(r.model, k)
    }
    return [...by.entries()].map(([model, v]) => ({ model, n: v.n, mean: v.total / v.n, perSample: v.perSample / v.n }))
  }, [rows])

  const completed = rows?.filter((r) => r.status === 'completed').length ?? 0
  const failed = rows?.filter((r) => r.status === 'failed').length ?? 0
  const available = models.filter((m) => m.available).length

  return (
    <>
      <PageHeader
        title="Overview"
        description="State of the official NCM checkpoints and the experiments recorded on this machine."
        actions={
          <Button variant="outline" size="sm" onClick={() => { void refresh(); void reload() }}>
            <RefreshCw /> Refresh
          </Button>
        }
      />
      <div className="space-y-6 p-4 sm:p-6 lg:p-8">
        {(status === 'error' || error) && (
          <div className="rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            {error ?? useModels.getState().error}
          </div>
        )}

        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <StatTile label="Checkpoints available" value={`${available}/${models.length || '—'}`} hint={`device · ${device ?? '—'}`} />
          <StatTile
            label="Experiments recorded"
            value={rows === null ? '—' : rows.length}
            hint={rows && rows.length === LIMIT ? `showing latest ${LIMIT}` : 'SQLite · results/playground.sqlite'}
          />
          <StatTile label="Completed / failed" value={rows === null ? '—' : `${completed} / ${failed}`} />
          <StatTile
            label="Mean latency per sample"
            value={latency.length ? fmtSeconds(latency.reduce((a, l) => a + l.perSample * l.n, 0) / latency.reduce((a, l) => a + l.n, 0)) : '—'}
            hint="generate + decode, across all models; per-model below"
          />
        </div>

        <div className="grid gap-6 xl:grid-cols-5">
          <Panel title="Models" meta="official HazyResearch checkpoints" className="xl:col-span-3" bodyClassName="p-0">
            <div className="divide-y divide-border">
              {models.map((m) => (
                <div key={m.id} className="flex flex-wrap items-center gap-x-4 gap-y-1 px-4 py-3">
                  <StatusDot tone={!m.available ? 'error' : m.loaded ? 'ok' : 'idle'} />
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                      <span className="font-mono text-[0.82rem] whitespace-nowrap">{m.id}</span>
                      <Badge variant="outline" className="font-normal">{m.conditional ? `prompt · ${m.prompt_kind}` : 'unconditional'}</Badge>
                    </div>
                    <div className="mt-0.5 truncate text-xs text-muted-foreground">{m.description}</div>
                  </div>
                  <div className="flex w-full gap-4 pl-3.5 font-mono text-xs text-muted-foreground tabular sm:w-auto sm:pl-0">
                    <span title="levels">{m.levels.length} lv</span>
                    <span title="output tokens">{m.output_tokens} tok</span>
                    <span title="generator parameters">
                      {m.parameters ? fmtParams(m.parameters.generator) : m.available ? 'not loaded' : 'missing'}
                    </span>
                  </div>
                </div>
              ))}
              {models.length === 0 && <div className="px-4 py-6 text-sm text-muted-foreground">No model info yet.</div>}
            </div>
          </Panel>

          <Panel title="Latency by model" meta="completed experiments" className="xl:col-span-2" bodyClassName="p-0">
            {latency.length === 0 ? (
              <div className="p-4"><EmptyState icon={Inbox} title="No completed experiments">Run one in the Playground.</EmptyState></div>
            ) : (
              <div className="overflow-x-auto"><table className="w-full text-sm">
                <thead className="text-xs text-muted-foreground">
                  <tr className="border-b border-border">
                    <th className="px-3 py-2 text-left font-normal sm:px-4">model</th>
                    <th className="px-3 py-2 text-right font-normal sm:px-4">runs</th>
                    <th className="px-3 py-2 text-right font-normal sm:px-4">per run</th>
                    <th className="px-3 py-2 text-right font-normal sm:px-4">per sample</th>
                  </tr>
                </thead>
                <tbody className="font-mono text-[0.8rem] tabular">
                  {latency.map((l) => (
                    <tr key={l.model} className="border-b border-border last:border-0">
                      <td className="px-3 py-2 sm:px-4">{l.model}</td>
                      <td className="px-3 py-2 text-right sm:px-4">{l.n}</td>
                      <td className="px-3 py-2 text-right sm:px-4">{fmtSeconds(l.mean)}</td>
                      <td className="px-3 py-2 text-right sm:px-4">{fmtSeconds(l.perSample)}</td>
                    </tr>
                  ))}
                </tbody>
              </table></div>
            )}
            <p className="border-t border-border px-4 py-2.5 text-xs text-muted-foreground">
              Ad-hoc runs with mixed batch sizes and settings: not a benchmark. Controlled measurements live in Benchmarks.
            </p>
          </Panel>
        </div>

        <div className="grid gap-6 xl:grid-cols-5">
          <Panel title="Latest experiments" className="xl:col-span-3" bodyClassName="p-0">
            {rows && rows.length > 0 ? (
              <div className="divide-y divide-border">
                {rows.slice(0, 8).map((r) => (
                  <div key={r.id} className="flex items-center gap-2 px-3 py-2.5 text-sm sm:gap-3 sm:px-4">
                    <StatusDot tone={r.status === 'completed' ? 'ok' : r.status === 'failed' ? 'error' : 'busy'} />
                    <span className="hidden w-16 shrink-0 text-xs text-muted-foreground sm:inline">{fmtRelative(r.created_at)}</span>
                    <span className="w-24 shrink-0 truncate font-mono text-[0.8rem] sm:w-32">{r.model}</span>
                    <span className="min-w-0 flex-1 truncate text-muted-foreground">
                      {r.status === 'failed' ? r.error : r.request.prompt ?? (r.request.example_index !== undefined && r.request.example_index !== null ? `test example #${r.request.example_index}` : `seed ${r.request.seed} · ${r.request.num_samples} samples`)}
                    </span>
                    <span className="shrink-0 font-mono text-xs text-muted-foreground tabular">{fmtSeconds(r.timing?.total_s)}</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="p-4">
                <EmptyState icon={FlaskConical} title="No experiments yet">
                  <Link to="/playground" className="text-primary hover:underline">Open the Playground</Link> to run NCM.
                </EmptyState>
              </div>
            )}
          </Panel>
          <Panel title="Recent benchmark results" className="xl:col-span-2">
            <EmptyState icon={BarChart3} title="No benchmark runs">
              The benchmarking workflow (warm-up, repeats, percentiles) is Milestone 10. Nothing is shown until real runs exist.
            </EmptyState>
          </Panel>
        </div>
      </div>
    </>
  )
}
