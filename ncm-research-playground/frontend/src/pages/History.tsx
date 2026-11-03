import { ArrowRight, Download, History as HistoryIcon, RefreshCw, Repeat, Search } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'

import { EmptyState, KV, PageHeader, Panel, StatusDot } from '@/components/common'
import { describeInput } from '@/components/ExperimentPicker'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Replay } from '@/features/history/Replay'
import { api, csvUrl } from '@/lib/api'
import { fmtRelative, fmtSeconds } from '@/lib/format'
import type { ExperimentRecord } from '@/lib/types'
import { cn } from '@/lib/utils'
import { useModels } from '@/store/models'

const ALL = '__all__'

export function History() {
  const models = useModels((s) => s.models)
  const [rows, setRows] = useState<ExperimentRecord[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [model, setModel] = useState(ALL)
  const [status, setStatus] = useState(ALL)
  const [q, setQ] = useState('')
  const [openId, setOpenId] = useState<string | null>(null)
  const [detail, setDetail] = useState<ExperimentRecord | null>(null)
  const [rerun, setRerun] = useState<{ from: string; id?: string; error?: string } | null>(null)

  const load = useCallback(() => api.experiments(500).then(setRows, (e: Error) => setError(e.message)), [])
  useEffect(() => { void load() }, [load])
  useEffect(() => {
    if (!openId) return
    let live = true
    api.experiment(openId).then((r) => live && setDetail(r), (e: Error) => live && setError(e.message))
    return () => { live = false }
  }, [openId])

  const filtered = useMemo(() => (rows ?? []).filter((r) =>
    (model === ALL || r.model === model) && (status === ALL || r.status === status) &&
    (!q || `${r.id} ${r.model} ${r.request.prompt ?? ''} ${r.request.tag ?? ''} ${(r.outputs ?? []).join(' ')}`.toLowerCase().includes(q.toLowerCase()))),
  [rows, model, status, q])

  const open = detail && detail.id === openId ? detail : null
  const info = models.find((m) => m.id === open?.model)

  const doRerun = async (rec: ExperimentRecord) => {
    setRerun({ from: rec.id })
    try {
      const { id } = await api.createExperiment(rec.request)
      setRerun({ from: rec.id, id })
      setTimeout(() => void load(), 500)
    } catch (e) {
      setRerun({ from: rec.id, error: (e as Error).message })
    }
  }

  return (
    <>
      <PageHeader
        title="Experiment History"
        description="Every run recorded in SQLite: its request, outputs, metrics and the execution events it emitted."
        actions={
          <>
            <Button variant="outline" size="sm" onClick={() => void load()}><RefreshCw />Refresh</Button>
            <Button variant="outline" size="sm" nativeButton={false} render={<a href={csvUrl.experiments()} download />}><Download />experiments.csv</Button>
          </>
        }
      />
      <div className="grid grid-cols-1 gap-6 p-4 sm:p-6 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)] lg:p-8">
        <Panel title="Runs" meta={rows ? `${filtered.length} of ${rows.length}` : undefined} bodyClassName="p-0">
          <div className="flex flex-wrap gap-2 border-b border-border p-3">
            <div className="relative min-w-40 flex-1">
              <Search className="pointer-events-none absolute top-2 left-2 size-4 text-muted-foreground" />
              <Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="search prompt, output, tag, id" className="h-8 pl-8 text-xs" />
            </div>
            <Select value={model} onValueChange={(v) => setModel((v as string) ?? ALL)}>
              <SelectTrigger size="sm" className="font-mono text-xs"><SelectValue>{model === ALL ? 'all models' : model}</SelectValue></SelectTrigger>
              <SelectContent>
                <SelectItem value={ALL} className="text-xs">all models</SelectItem>
                {models.map((m) => <SelectItem key={m.id} value={m.id} className="font-mono text-xs">{m.id}</SelectItem>)}
              </SelectContent>
            </Select>
            <Select value={status} onValueChange={(v) => setStatus((v as string) ?? ALL)}>
              <SelectTrigger size="sm" className="text-xs"><SelectValue>{status === ALL ? 'any status' : status}</SelectValue></SelectTrigger>
              <SelectContent>
                {[ALL, 'completed', 'failed', 'running', 'queued'].map((s) => <SelectItem key={s} value={s} className="text-xs">{s === ALL ? 'any status' : s}</SelectItem>)}
              </SelectContent>
            </Select>
          </div>
          {error && <div className="p-3 text-sm text-destructive">{error}</div>}
          {rows?.length === 0 && <div className="p-4"><EmptyState icon={HistoryIcon} title="No experiments recorded yet" /></div>}
          <div className="max-h-[70vh] divide-y divide-border overflow-y-auto">
            {filtered.map((r) => (
              <button key={r.id} type="button" onClick={() => setOpenId(r.id)}
                className={cn('flex w-full items-center gap-3 px-3 py-2 text-left text-sm', r.id === openId ? 'bg-primary/10' : 'hover:bg-muted/40')}>
                <StatusDot tone={r.status === 'completed' ? 'ok' : r.status === 'failed' ? 'error' : 'busy'} />
                <span className="hidden w-14 shrink-0 text-xs text-muted-foreground sm:inline">{fmtRelative(r.created_at)}</span>
                <span className={cn('w-36 shrink-0 truncate font-mono text-[0.78rem]', r.model.startsWith('ar-') ? 'text-ar' : 'text-ncm')}>{r.model}</span>
                <span className="min-w-0 flex-1 truncate text-xs text-muted-foreground">{r.status === 'failed' ? r.error : describeInput(r)}</span>
                {r.request.tag && <Badge variant="outline" className="hidden font-mono text-[0.65rem] font-normal md:inline-flex">{r.request.tag}</Badge>}
                <span className="shrink-0 font-mono text-xs text-muted-foreground tabular">{fmtSeconds(r.timing?.total_s)}</span>
              </button>
            ))}
          </div>
        </Panel>

        <div className="min-w-0 space-y-4">
          {!open ? (
            <EmptyState icon={HistoryIcon} title="Select a run">Reopen its outputs and metrics, replay its recorded events, or run the same request again.</EmptyState>
          ) : (
            <>
              <Panel title={<span className="font-mono normal-case">{open.model}</span>} meta={<span className="font-mono">{open.id.slice(0, 8)} · {new Date(open.created_at).toLocaleString()}</span>}>
                <div className="flex flex-wrap gap-2">
                  {open.request.include_trace !== false && open.status === 'completed' && !open.model.startsWith('ar-') && (
                    <>
                      <Button variant="outline" size="sm" nativeButton={false} render={<Link to={`/concepts?id=${open.id}`} />}>Concept Explorer <ArrowRight /></Button>
                      <Button variant="outline" size="sm" nativeButton={false} render={<Link to={`/timeline?id=${open.id}`} />}>Timeline <ArrowRight /></Button>
                    </>
                  )}
                  <Button variant="outline" size="sm" onClick={() => void doRerun(open)} disabled={rerun?.from === open.id && !rerun.id && !rerun.error}>
                    <Repeat />Re-run same request
                  </Button>
                </div>
                {rerun?.from === open.id && (
                  <p className="mt-2 text-xs text-muted-foreground">
                    {rerun.error ? <span className="text-destructive">{rerun.error}</span>
                      : rerun.id ? <>Started a new experiment <button type="button" className="font-mono text-primary hover:underline" onClick={() => setOpenId(rerun.id!)}>{rerun.id.slice(0, 8)}</button> (new measurement, same request).</>
                        : 'submitting…'}
                  </p>
                )}
                <div className="mt-4 grid grid-cols-1 gap-x-6 md:grid-cols-2">
                  <div>
                    <KV k="status" v={open.status} />
                    <KV k="input" v={describeInput(open)} />
                    <KV k="samples · seed" v={`${open.request.num_samples} · ${open.request.seed}`} />
                    <KV k="sampling" v={open.metrics ? `T=${open.metrics.sampling.temperature} k=${open.metrics.sampling.top_k} p=${open.metrics.sampling.top_p}` : '—'} />
                    {open.request.tag && <KV k="tag" v={open.request.tag} />}
                  </div>
                  <div>
                    <KV k="generate" v={fmtSeconds(open.timing?.generate_s)} />
                    <KV k="total (excl. load)" v={fmtSeconds(open.timing?.total_s)} />
                    <KV k="checkpoint load" v={fmtSeconds(open.timing?.load_s)} />
                    <KV k="forward passes" v={open.metrics?.transformer_passes ?? '—'} />
                    <KV k="gen-PPL" v={open.metrics?.quality?.corpus_ppl?.toFixed(2) ?? 'not scored'} />
                  </div>
                </div>
                {open.error && <p className="mt-3 text-sm text-destructive">{open.error}</p>}
              </Panel>

              {open.events && open.events.length > 0 && (
                <Panel title="Replay" meta="recorded events, not a new run">
                  <Replay key={open.id} events={open.events} model={info} />
                </Panel>
              )}

              {open.outputs && (
                <Panel title="Outputs">
                  <div className="space-y-3">
                    {open.outputs.map((o, i) => (
                      <div key={i} className="rounded-lg border border-border bg-background/40 p-3">
                        <div className="mb-1 font-mono text-[0.68rem] text-muted-foreground">sample {i}</div>
                        <p className="text-[0.84rem] leading-relaxed whitespace-pre-wrap">{o.replace(/^\s+/, '') || <span className="text-muted-foreground">(empty)</span>}</p>
                      </div>
                    ))}
                    {open.metrics?.reference && <p className="text-xs text-muted-foreground"><span className="text-foreground">reference:</span> {open.metrics.reference}</p>}
                  </div>
                </Panel>
              )}
            </>
          )}
        </div>
      </div>
    </>
  )
}
