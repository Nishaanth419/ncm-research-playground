import { AlertTriangle, Dices, GitCompareArrows, Loader2, Play } from 'lucide-react'
import type { ReactNode } from 'react'
import { useEffect, useMemo } from 'react'
import { Link } from 'react-router-dom'

import { EmptyState, PageHeader, Panel, StatusDot } from '@/components/common'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { StepProgress } from '@/features/compare/StepProgress'
import { ParamControl } from '@/features/playground/ParamControl'
import { LevelStrip } from '@/features/playground/LevelStrip'
import { caveats, summarize, type Summary } from '@/lib/compare'
import { fmtInt, fmtParams, fmtSeconds } from '@/lib/format'
import type { ModelInfo } from '@/lib/types'
import { cn } from '@/lib/utils'
import { useComparison, type Side, type SideState } from '@/store/comparison'
import { useModels } from '@/store/models'

const STATUS: Record<SideState['status'], string> = {
  idle: 'idle', queued: 'queued', 'loading-model': 'loading checkpoint…', running: 'generating…',
  scoring: 'scoring with GPT-2-large…', done: 'done', failed: 'failed',
}

const num = (v: number | null, d = 2) => (v === null ? '—' : v.toFixed(d))

type Row = { label: string; hint?: string; fmt: (s: Summary) => string; better?: 'lower' | 'higher'; key?: keyof Summary; comparable?: boolean }

const ROWS: { group: string; rows: Row[] }[] = [
  { group: 'Latency (one run)', rows: [
    { label: 'total', hint: 'generate + decode, excludes checkpoint load', fmt: (s) => fmtSeconds(s.totalS), key: 'totalS', better: 'lower' },
    { label: 'generate', fmt: (s) => fmtSeconds(s.generateS), key: 'generateS', better: 'lower' },
    { label: 'first output', hint: 'AR: first token · NCM: complete text', fmt: (s) => fmtSeconds(s.firstOutputS), comparable: false },
    { label: 'sequential forward passes', fmt: (s) => fmtInt(s.passes), key: 'passes', better: 'lower' },
  ] },
  { group: 'Throughput & length', rows: [
    { label: 'words / s', hint: 'whitespace words, comparable across tokenizers', fmt: (s) => num(s.wordsPerS, 1), key: 'wordsPerS', better: 'higher' },
    { label: 'tokens / s', hint: "own tokenizer: not comparable", fmt: (s) => num(s.tokensPerS, 1), comparable: false },
    { label: 'mean output words', fmt: (s) => num(s.meanWords, 1), comparable: false },
    { label: 'mean output tokens', hint: 'own tokenizer', fmt: (s) => num(s.meanTokens, 1), comparable: false },
    { label: 'device memory', fmt: (s) => (s.memoryBytes === null ? 'n/a' : `${(s.memoryBytes / 2 ** 20).toFixed(0)} MiB`), comparable: false },
  ] },
  { group: 'Quality', rows: [
    { label: 'gen-PPL (GPT-2-large)', hint: 'official NCM protocol; lower = more fluent', fmt: (s) => num(s.genPpl), key: 'genPpl', better: 'lower' },
    { label: 'entropy, judge tokens', hint: 'nats; low = repetitive', fmt: (s) => num(s.entropyJudge, 3), comparable: false },
    { label: 'entropy, own tokens', hint: 'as in eval_merge.py; not comparable', fmt: (s) => num(s.entropyOwn, 3), comparable: false },
  ] },
]

function ModelCard({ info, side }: { info?: ModelInfo; side: Side }) {
  if (!info) return null
  const params = info.parameters
  return (
    <div className={cn('rounded-lg border bg-background/40 p-3', side === 'ncm' ? 'border-ncm/30' : 'border-ar/30')}>
      <div className={cn('font-mono text-[0.82rem]', side === 'ncm' ? 'text-ncm' : 'text-ar')}>{info.id}</div>
      <div className="mt-1 text-xs leading-relaxed text-muted-foreground">{info.description}</div>
      <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs">
        <dt className="text-muted-foreground">params</dt>
        <dd className="font-mono">{params ? Object.entries(params).map(([k, v]) => `${k} ${fmtParams(v)}`).join(' + ') : 'known after first load'}</dd>
        <dt className="text-muted-foreground">tokenizer</dt><dd className="font-mono">{info.tokenizer}</dd>
        <dt className="text-muted-foreground">data</dt><dd>{info.training_data}</dd>
        <dt className="text-muted-foreground">length</dt><dd>{info.notes.find((n) => n.includes('token')) ?? `${info.output_tokens} tokens`}</dd>
      </dl>
    </div>
  )
}

function Progress({ side, state, info }: { side: Side; state: SideState; info?: ModelInfo }) {
  return (
    <div className="min-w-0">
      <div className="mb-2 flex items-center gap-2 text-sm">
        <StatusDot tone={state.status === 'done' ? 'ok' : state.status === 'failed' ? 'error' : state.status === 'idle' ? 'idle' : 'busy'} />
        <span className={cn('font-mono text-[0.8rem]', side === 'ncm' ? 'text-ncm' : 'text-ar')}>{info?.id}</span>
        <span className="text-muted-foreground">{STATUS[state.status]}</span>
      </div>
      {side === 'ncm' && info
        ? <LevelStrip schedule={info.levels} levels={state.levels} />
        : <StepProgress steps={state.steps} max={info?.output_tokens ?? 128} done={state.status === 'done' || state.status === 'scoring'} />}
      {state.error && <p className="mt-2 text-xs text-destructive">{state.error}</p>}
    </div>
  )
}

export function Comparison() {
  const { models, refresh } = useModels()
  const st = useComparison()
  const ncmInfo = models.find((m) => m.id === st.models.ncm)
  const arInfo = models.find((m) => m.id === st.models.ar)
  const busy = st.phase === 'running' || st.phase === 'scoring'
  const ready = !!ncmInfo?.available && !!arInfo?.available

  // parameter counts are known once both checkpoints are loaded
  useEffect(() => {
    if (st.phase === 'done') void refresh()
  }, [st.phase, refresh])

  const mN = st.sides.ncm.record?.metrics, mA = st.sides.ar.record?.metrics
  const sN = mN ? summarize(mN) : null, sA = mA ? summarize(mA) : null
  const notes = useMemo(() => (ncmInfo && arInfo && mN && mA ? caveats(ncmInfo, arInfo, mN, mA) : []), [ncmInfo, arInfo, mN, mA])

  const winner = (r: Row): Side | null => {
    if (!sN || !sA || !r.key || !r.better) return null
    const a = sN[r.key] as number | null, b = sA[r.key] as number | null
    if (a === null || b === null || a === b) return null
    return (r.better === 'lower' ? a < b : a > b) ? 'ncm' : 'ar'
  }

  return (
    <>
      <PageHeader
        title="Model Comparison"
        description="NCM and a token-by-token autoregressive baseline, run back to back on the same machine with the same seed, batch and sampling settings."
      />
      <div className="space-y-6 p-4 sm:p-6 lg:p-8">
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-[360px_minmax(0,1fr)]">
          <div className="space-y-4">
            <Panel title="Models">
              <div className="space-y-3">
                <ModelCard info={ncmInfo} side="ncm" />
                <ModelCard info={arInfo} side="ar" />
              </div>
            </Panel>
            <Panel title="Shared settings" meta="applied to both">
              <div className="space-y-4">
                <ParamControl label="temperature" value={st.sampling.temperature} defaultValue={1} min={0} max={2} step={0.05}
                  onChange={(v) => st.set({ sampling: { ...st.sampling, temperature: v ?? 1 } })} />
                <ParamControl label="top-k" hint="0 = off" value={st.sampling.top_k} defaultValue={0} min={0} max={256} step={1}
                  onChange={(v) => st.set({ sampling: { ...st.sampling, top_k: v ?? 0 } })} />
                <ParamControl label="top-p" value={st.sampling.top_p} defaultValue={1} min={0} max={1} step={0.01}
                  onChange={(v) => st.set({ sampling: { ...st.sampling, top_p: v ?? 1 } })} />
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1.5">
                    <Label className="text-[0.8rem] font-normal">seed</Label>
                    <div className="flex gap-1">
                      <Input type="number" min={0} value={st.seed} onChange={(e) => st.set({ seed: Math.max(0, Number(e.target.value) || 0) })} className="h-8 font-mono text-xs" />
                      <Button variant="outline" size="icon" title="random seed" onClick={() => st.set({ seed: Math.floor(Math.random() * 2 ** 31) })}><Dices /></Button>
                    </div>
                  </div>
                  <div className="space-y-1.5">
                    <Label className="text-[0.8rem] font-normal">samples</Label>
                    <Input type="number" min={1} max={16} value={st.numSamples} onChange={(e) => st.set({ numSamples: Math.max(1, Math.min(16, Number(e.target.value) || 1)) })} className="h-8 font-mono text-xs" />
                  </div>
                </div>
                <p className="text-xs text-muted-foreground">Both checkpoints’ own defaults are pure sampling (T=1, no top-k/top-p). A shared seed does not give the same random draws across different models.</p>
              </div>
            </Panel>
            <Button size="lg" className="h-10 w-full" disabled={!ready || busy} onClick={() => void st.run()}>
              {busy ? <Loader2 className="animate-spin" /> : <Play />}
              {st.phase === 'running' ? 'Generating' : st.phase === 'scoring' ? 'Scoring quality' : 'Run comparison'}
            </Button>
            {!ready && models.length > 0 && <p className="text-xs text-destructive">A checkpoint is missing; see Overview.</p>}
          </div>

          <div className="min-w-0 space-y-4">
            <Panel title="Execution" meta={st.tag && <span className="font-mono">{st.tag}</span>}>
              <div className="space-y-5">
                <Progress side="ncm" state={st.sides.ncm} info={ncmInfo} />
                <Progress side="ar" state={st.sides.ar} info={arInfo} />
              </div>
              <p className="mt-4 text-xs text-muted-foreground">
                NCM: {ncmInfo?.levels.length ?? '…'} sequential levels, each predicting all its codes in one step. Baseline: one sequential step per token. The runs execute one after the other; each first does one untimed warm-up pass.
              </p>
            </Panel>

            {sN && sA ? (
              <Panel title="Measurements" bodyClassName="p-0">
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="text-xs">
                      <tr className="border-b border-border">
                        <th className="px-4 py-2 text-left font-normal text-muted-foreground">metric</th>
                        <th className="px-4 py-2 text-right font-mono font-normal text-ncm">{ncmInfo?.id}</th>
                        <th className="px-4 py-2 text-right font-mono font-normal text-ar">{arInfo?.id}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {ROWS.map((g) => (
                        <Group key={g.group} title={g.group}>
                          {g.rows.map((r) => {
                            const w = winner(r)
                            return (
                              <tr key={r.label} className="border-b border-border last:border-0">
                                <td className="px-4 py-1.5">
                                  <div>{r.label}</div>
                                  {r.hint && <div className="text-[0.7rem] text-muted-foreground">{r.hint}</div>}
                                </td>
                                {(['ncm', 'ar'] as Side[]).map((side) => (
                                  <td key={side} className={cn('px-4 py-1.5 text-right font-mono text-[0.8rem] tabular', r.comparable === false && 'text-muted-foreground', w === side && (side === 'ncm' ? 'text-ncm' : 'text-ar'))}>
                                    {r.fmt(side === 'ncm' ? sN : sA)}
                                  </td>
                                ))}
                              </tr>
                            )
                          })}
                        </Group>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="border-t border-border px-4 py-2.5 text-xs text-muted-foreground">
                  Coloured values are better on that row; grey rows are shown for context only and are not like-for-like across models.
                  {sN.memoryBytes === null && ` Memory: ${sN.memoryNote}.`}
                </p>
              </Panel>
            ) : (
              <EmptyState icon={GitCompareArrows} title={busy ? 'Waiting for both runs…' : 'No comparison yet'}>
                {busy ? 'Measurements appear once both models finish.' : 'Press Run comparison. Each model runs once with the shared settings, then both outputs are scored by GPT-2-large.'}
              </EmptyState>
            )}

            {notes.length > 0 && (
              <Panel title={<span className="flex items-center gap-1.5"><AlertTriangle className="size-3.5 text-primary" />Not like-for-like</span>}>
                <ul className="list-disc space-y-1.5 pl-4 text-[0.82rem] text-muted-foreground">
                  {notes.map((n) => <li key={n}>{n}</li>)}
                </ul>
              </Panel>
            )}

            {(st.sides.ncm.record || st.sides.ar.record) && (
              <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
                {(['ncm', 'ar'] as Side[]).map((side) => {
                  const rec = st.sides[side].record
                  const q = rec?.metrics?.quality
                  return (
                    <Panel key={side} title={<span className={side === 'ncm' ? 'text-ncm' : 'text-ar'}>{rec?.model ?? (side === 'ncm' ? ncmInfo?.id : arInfo?.id)}</span>}
                      meta={rec && side === 'ncm' ? <Link to={`/concepts?id=${rec.id}`} className="text-primary hover:underline">concepts →</Link> : undefined}>
                      <div className="space-y-3">
                        {(rec?.outputs ?? []).map((o, i) => (
                          <div key={i} className="rounded-lg border border-border bg-background/40 p-3">
                            <div className="mb-1 flex justify-between font-mono text-[0.68rem] text-muted-foreground">
                              <span>sample {i}</span>
                              <span>{q?.per_sample[i] ? `ppl ${q.per_sample[i]!.ppl.toFixed(1)}` : q ? 'not scored (empty)' : ''}</span>
                            </div>
                            <p className="text-[0.84rem] leading-relaxed whitespace-pre-wrap">{o.replace(/^\s+/, '') || <span className="text-muted-foreground">(empty)</span>}</p>
                          </div>
                        ))}
                      </div>
                    </Panel>
                  )
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  )
}

function Group({ title, children }: { title: string; children: ReactNode }) {
  return (
    <>
      <tr className="border-b border-border bg-muted/30">
        <td colSpan={3} className="px-4 py-1 text-[0.68rem] tracking-wider text-muted-foreground uppercase">{title}</td>
      </tr>
      {children}
    </>
  )
}
