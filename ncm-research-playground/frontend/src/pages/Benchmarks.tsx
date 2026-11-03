import { BarChart3, Download, Loader2, Play } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, ErrorBar, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

import { EmptyState, KV, PageHeader, Panel, StatusDot } from '@/components/common'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { api, csvUrl } from '@/lib/api'
import { chartRows, medianRatio } from '@/lib/bench'
import { fmtRelative, fmtSeconds } from '@/lib/format'
import type { BenchmarkRecord, ModelInfo } from '@/lib/types'
import { cn } from '@/lib/utils'
import { useModels } from '@/store/models'

const axis = { fill: 'var(--muted-foreground)', fontSize: 11, fontFamily: 'var(--font-mono)' }
const BATCHES = [1, 2, 4, 8, 16]
const colorOf = (m?: ModelInfo | null) => (m?.family === 'autoregressive' ? 'var(--ar)' : 'var(--ncm)')

function useBenchmark(id: string | null) {
  const [rec, setRec] = useState<BenchmarkRecord | null>(null)
  useEffect(() => {
    if (!id) return
    let live = true
    let timer: ReturnType<typeof setTimeout>
    const tick = async () => {
      try {
        const r = await api.benchmark(id)
        if (!live) return
        setRec(r)
        if (r.status === 'queued' || r.status === 'running') timer = setTimeout(tick, 1000)
      } catch {
        if (live) timer = setTimeout(tick, 2000)
      }
    }
    void tick()
    return () => { live = false; clearTimeout(timer) }
  }, [id])
  return rec && rec.id === id ? rec : null
}

function MedianChart({ rec, stat, title, unit }: { rec: BenchmarkRecord; stat: 'total_s' | 'words_per_s'; title: string; unit: 's' | 'words/s' }) {
  const rows = chartRows(rec.summary ?? [], stat)
  const models = rec.config.models
  return (
    <div className="min-w-0">
      <div className="mb-1 text-xs text-muted-foreground">{title}</div>
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} margin={{ left: -8, right: 8, top: 8, bottom: 0 }}>
            <CartesianGrid stroke="var(--border)" vertical={false} />
            <XAxis dataKey="batch" tick={axis} tickLine={false} axisLine={false} />
            <YAxis tick={axis} tickLine={false} axisLine={false} width={56}
              tickFormatter={(v: number) => (unit === 's' ? fmtSeconds(v) : v.toFixed(0))} />
            <Tooltip
              cursor={{ fill: 'var(--muted)' }}
              contentStyle={{ background: 'var(--popover)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }}
              formatter={(v, name) => [unit === 's' ? fmtSeconds(v as number) : `${(v as number).toFixed(1)} words/s`, `${name} · median`]}
            />
            <Legend wrapperStyle={{ fontSize: 11, fontFamily: 'var(--font-mono)' }} />
            {models.map((m) => (
              <Bar key={m} dataKey={m} fill={colorOf(rec.models?.[m])} fillOpacity={0.75} radius={[3, 3, 0, 0]} isAnimationActive={false}>
                <ErrorBar dataKey={`${m}__err`} width={6} stroke="var(--foreground)" strokeOpacity={0.6} direction="y" />
              </Bar>
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

export function Benchmarks() {
  const models = useModels((s) => s.models)
  const eligible = models.filter((m) => !m.conditional)
  const [list, setList] = useState<BenchmarkRecord[] | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [cfg, setCfg] = useState({ models: ['ncm-tinystories', 'ar-tinystories-33m'], batch_sizes: [1, 4], repeats: 10, warmup_runs: 2, seed: 0 })
  const [error, setError] = useState<string | null>(null)
  const rec = useBenchmark(selected)

  const loadList = useCallback(() => api.benchmarks(50).then((l) => {
    setList(l)
    setSelected((s) => s ?? l[0]?.id ?? null)
  }, (e: Error) => setError(e.message)), [])
  useEffect(() => { void loadList() }, [loadList])
  useEffect(() => { if (rec?.status === 'completed' || rec?.status === 'failed') void loadList() }, [rec?.status, loadList])

  const running = rec?.status === 'queued' || rec?.status === 'running'
  const totalRuns = cfg.models.length * cfg.batch_sizes.length * (cfg.repeats + cfg.warmup_runs)
  const toggle = <T,>(xs: T[], x: T) => (xs.includes(x) ? xs.filter((y) => y !== x) : [...xs, x])

  const start = async () => {
    setError(null)
    try {
      const { id } = await api.createBenchmark({ name: '', sampling: {}, ...cfg, batch_sizes: [...cfg.batch_sizes].sort((a, b) => a - b) })
      setSelected(id)
      void loadList()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const ncmId = rec?.config.models.find((m) => rec.models?.[m]?.family === 'ncm')
  const arId = rec?.config.models.find((m) => rec.models?.[m]?.family === 'autoregressive')

  return (
    <>
      <PageHeader
        title="Benchmarks"
        description="Repeated, warmed-up measurements on this machine. Medians and percentiles, hardware and configuration recorded with every run."
        actions={list && list.length > 0 && (
          <Select value={selected ?? ''} onValueChange={(v) => v && setSelected(v as string)}>
            <SelectTrigger className="w-[min(22rem,calc(100vw-6rem))] font-mono text-[0.78rem]">
              <SelectValue>{rec ? `${rec.config.models.length} models · ×${rec.config.repeats} · ${fmtRelative(rec.created_at)}` : 'benchmark'}</SelectValue>
            </SelectTrigger>
            <SelectContent>
              {list.map((b) => (
                <SelectItem key={b.id} value={b.id} className="font-mono text-[0.78rem]">
                  {b.status} · {b.config.models.join(' vs ')} · b{b.config.batch_sizes.join('/')} · ×{b.config.repeats} · {fmtRelative(b.created_at)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}
      />
      <div className="grid grid-cols-1 gap-6 p-4 sm:p-6 lg:grid-cols-[320px_minmax(0,1fr)] lg:p-8">
        <div className="space-y-4">
          <Panel title="New benchmark">
            <div className="space-y-4">
              <div>
                <Label className="mb-1.5 block text-[0.8rem] font-normal">models</Label>
                <div className="space-y-1">
                  {eligible.map((m) => (
                    <label key={m.id} className="flex cursor-pointer items-center gap-2 rounded-md px-1.5 py-1 hover:bg-muted/50">
                      <input type="checkbox" className="accent-[var(--ncm)]" checked={cfg.models.includes(m.id)} disabled={!m.available}
                        onChange={() => setCfg((c) => ({ ...c, models: toggle(c.models, m.id).slice(0, 4) }))} />
                      <span className="font-mono text-[0.78rem]" style={{ color: colorOf(m) }}>{m.id}</span>
                    </label>
                  ))}
                </div>
                <p className="mt-1 text-xs text-muted-foreground">Unconditional models only: the workload has no prompts.</p>
              </div>
              <div>
                <Label className="mb-1.5 block text-[0.8rem] font-normal">batch sizes (max 4)</Label>
                <div className="flex gap-1">
                  {BATCHES.map((b) => (
                    <button key={b} type="button"
                      onClick={() => setCfg((c) => ({ ...c, batch_sizes: c.batch_sizes.includes(b) ? c.batch_sizes.filter((x) => x !== b) : c.batch_sizes.length < 4 ? [...c.batch_sizes, b] : c.batch_sizes }))}
                      className={cn('h-7 w-9 rounded-md border font-mono text-xs', cfg.batch_sizes.includes(b) ? 'border-primary/60 bg-primary/15 text-primary' : 'border-border text-muted-foreground hover:bg-muted/50')}>
                      {b}
                    </button>
                  ))}
                </div>
              </div>
              <div className="grid grid-cols-3 gap-2">
                {([['repeats', 1, 100], ['warmup_runs', 0, 5], ['seed', 0, 2 ** 31 - 1]] as const).map(([k, lo, hi]) => (
                  <div key={k} className="space-y-1.5">
                    <Label className="text-[0.75rem] font-normal">{k === 'warmup_runs' ? 'warm-ups' : k}</Label>
                    <Input type="number" min={lo} max={hi} value={cfg[k]}
                      onChange={(e) => setCfg((c) => ({ ...c, [k]: Math.max(lo, Math.min(hi, Number(e.target.value) || 0)) }))}
                      className="h-8 font-mono text-xs" />
                  </div>
                ))}
              </div>
              <p className="text-xs text-muted-foreground">
                {totalRuns} generations ({cfg.warmup_runs} untimed warm-ups + {cfg.repeats} timed per model × batch). Timed run i uses seed {cfg.seed}+i. Checkpoint defaults for sampling.
              </p>
            </div>
          </Panel>
          <Button size="lg" className="h-10 w-full" disabled={running || cfg.models.length === 0 || cfg.batch_sizes.length === 0} onClick={() => void start()}>
            {running ? <Loader2 className="animate-spin" /> : <Play />} {running ? 'Running' : 'Run benchmark'}
          </Button>
          {error && <p className="text-xs text-destructive">{error}</p>}
        </div>

        <div className="min-w-0 space-y-4">
          {!rec && list?.length === 0 && (
            <EmptyState icon={BarChart3} title="No benchmarks yet">Configure one on the left. Results are stored in SQLite and exportable as CSV.</EmptyState>
          )}
          {rec && (
            <>
              <Panel title="Status" meta={<span className="font-mono">{rec.id.slice(0, 8)}</span>}>
                <div className="flex items-center gap-2 text-sm">
                  <StatusDot tone={rec.status === 'completed' ? 'ok' : rec.status === 'failed' ? 'error' : 'busy'} />
                  <span>{rec.status}</span>
                  {rec.progress && running && (
                    <span className="text-muted-foreground">
                      · {rec.progress.phase === 'warmup' ? `warm-up ${rec.progress.warmup}` : rec.progress.phase} {rec.progress.model ?? ''} {rec.progress.batch_size ? `batch ${rec.progress.batch_size}` : ''}
                    </span>
                  )}
                  {rec.error && <span className="text-destructive">· {rec.error}</span>}
                </div>
                {rec.progress && (
                  <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-muted">
                    <div className="h-full bg-primary transition-[width]" style={{ width: `${(rec.progress.done / Math.max(1, rec.progress.total)) * 100}%` }} />
                  </div>
                )}
                <div className="mt-1.5 font-mono text-[0.7rem] text-muted-foreground">{rec.progress?.done ?? 0} / {rec.progress?.total ?? '—'} timed runs</div>
              </Panel>

              {(rec.summary?.length ?? 0) > 0 && (
                <>
                  <Panel title="Results" meta="median · p90 · p95 over timed runs" bodyClassName="p-0">
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead className="text-xs text-muted-foreground">
                          <tr className="border-b border-border">
                            {['model', 'batch', 'n', 'total median', 'p90', 'p95', 'per sample', 'first output', 'words/s', 'passes'].map((h, i) => (
                              <th key={h} className={cn('px-3 py-2 font-normal whitespace-nowrap', i < 2 ? 'text-left' : 'text-right')}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody className="font-mono text-[0.78rem] whitespace-nowrap tabular">
                          {rec.summary!.map((g) => (
                            <tr key={`${g.model}-${g.batch_size}`} className="border-b border-border last:border-0">
                              <td className="px-3 py-1.5 whitespace-nowrap" style={{ color: colorOf(rec.models?.[g.model]) }}>{g.model}</td>
                              <td className="px-3">{g.batch_size}</td>
                              <td className="px-3 text-right">{g.n}</td>
                              <td className="px-3 text-right">{fmtSeconds(g.total_s.median)}</td>
                              <td className="px-3 text-right text-muted-foreground">{fmtSeconds(g.total_s.p90)}</td>
                              <td className="px-3 text-right text-muted-foreground">{fmtSeconds(g.total_s.p95)}</td>
                              <td className="px-3 text-right">{fmtSeconds(g.per_sample_s.median)}</td>
                              <td className="px-3 text-right">{fmtSeconds(g.first_output_s.median)}</td>
                              <td className="px-3 text-right">{g.words_per_s.median?.toFixed(1) ?? '—'}</td>
                              <td className="px-3 text-right">{g.passes}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    {ncmId && arId && rec.status === 'completed' && (
                      <div className="border-t border-border px-4 py-2.5 text-xs text-muted-foreground">
                        {rec.config.batch_sizes.map((bs) => {
                          const r = medianRatio(rec.summary!, arId, ncmId, bs, 'total_s')
                          return r && <div key={bs}>batch {bs}: {arId} median total is <span className="font-mono text-foreground">{r.toFixed(2)}×</span> {ncmId}’s on this machine.</div>
                        })}
                      </div>
                    )}
                  </Panel>

                  <Panel title="Charts" meta="bar = median · whisker = p95">
                    <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
                      <MedianChart rec={rec} stat="total_s" title="total latency per run (generate + decode)" unit="s" />
                      <MedianChart rec={rec} stat="words_per_s" title="throughput (whitespace words / s)" unit="words/s" />
                    </div>
                  </Panel>
                </>
              )}

              <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
                <Panel title="Hardware">
                  {rec.hardware ? Object.entries(rec.hardware).map(([k, v]) => (
                    <KV key={k} k={k} v={k === 'ram_bytes' && typeof v === 'number' ? `${(v / 2 ** 30).toFixed(0)} GiB` : String(v ?? '—')} />
                  )) : <p className="text-sm text-muted-foreground">recorded when the benchmark starts</p>}
                </Panel>
                <Panel title="Configuration">
                  <KV k="models" v={rec.config.models.join(', ')} />
                  <KV k="batch sizes" v={rec.config.batch_sizes.join(', ')} />
                  <KV k="timed runs each" v={rec.config.repeats} />
                  <KV k="warm-ups each" v={rec.config.warmup_runs} />
                  <KV k="seeds" v={`${rec.config.seed} … ${rec.config.seed + rec.config.repeats - 1}`} />
                  <KV k="sampling" v="checkpoint defaults (T=1, no top-k / top-p)" />
                  <KV k="timing" v="device-synced, excludes load, no trace" />
                  <div className="mt-3 flex flex-wrap gap-2">
                    <Button variant="outline" size="sm" render={<a href={csvUrl.benchmarkSummary(rec.id)} download />}><Download />summary.csv</Button>
                    <Button variant="outline" size="sm" render={<a href={csvUrl.benchmarkRuns(rec.id)} download />}><Download />runs.csv</Button>
                  </div>
                </Panel>
              </div>

              <Panel title="Reading these numbers">
                <ul className="list-disc space-y-1.5 pl-4 text-[0.82rem] text-muted-foreground">
                  <li>Measured on {String(rec.hardware?.cpu ?? 'this machine')} ({String(rec.hardware?.device ?? '')}). The NCM paper’s speedups were measured on an H100 against diffusion models, not against this baseline; do not compare them with these numbers.</li>
                  <li>“first output” is the first token for autoregressive models and the complete text for NCM.</li>
                  <li>Words/s uses whitespace words so it is comparable across tokenizers. The NCM model always fills 128 tokens; the baseline stops at EOS, so its output length varies per run.</li>
                  <li>Parameter counts, tokenizers and training runs differ between models: see Model Comparison for the full list.</li>
                </ul>
              </Panel>
            </>
          )}
        </div>
      </div>
    </>
  )
}
