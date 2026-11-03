import { ArrowRight, FlaskConical } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

import { EmptyState, KV, PageHeader, Panel, StatusDot } from '@/components/common'
import { ExperimentPicker } from '@/components/ExperimentPicker'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { TimelineBar } from '@/features/timeline/TimelineBar'
import { useExperimentTrace } from '@/hooks/useExperimentTrace'
import { fmtSeconds } from '@/lib/format'
import { timelineSegments } from '@/lib/trace'
import { cn } from '@/lib/utils'

const axis = { fill: 'var(--muted-foreground)', fontSize: 10, fontFamily: 'var(--font-mono)' }

export function Timeline() {
  const { candidates, selected, select, trace, record, error } = useExperimentTrace()
  const [showLoad, setShowLoad] = useState(false)
  // reset the selected stage when the experiment changes (derived, no effect)
  const [picked, setPicked] = useState<{ id: string | null; level: number | null }>({ id: null, level: null })
  const level = picked.id === selected ? picked.level : null
  const setLevel = (l: number) => setPicked({ id: selected, level: l })

  const all = useMemo(() => timelineSegments(record?.events ?? []), [record])
  const segments = showLoad ? all : all.filter((s) => s.kind !== 'load')
  const load = all.find((s) => s.kind === 'load')
  const decode = all.find((s) => s.kind === 'decode')
  const lv = level !== null ? trace?.levels[level] : undefined
  const perLevel = (trace?.levels ?? []).map((l) => ({ level: `L${l.level}`, seconds: l.seconds ?? 0, n: l.n_codes, src: l.source }))

  return (
    <>
      <PageHeader
        title="Generation Timeline"
        description="Levels run one after another; within a level, all codes are predicted in a single step. Built from the recorded execution events of one experiment."
        actions={<ExperimentPicker candidates={candidates} selected={selected} onSelect={select} />}
      />
      <div className="space-y-6 p-4 sm:p-6 lg:p-8">
        {error && <div className="rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">{error}</div>}
        {candidates?.length === 0 && (
          <EmptyState icon={FlaskConical} title="No traced experiments yet">
            Run something in the <Link to="/playground" className="text-primary hover:underline">Playground</Link>.
          </EmptyState>
        )}
        {trace && record && (
          <>
            <Panel
              title="Execution"
              meta={
                load ? (
                  <div className="flex items-center gap-2">
                    <Switch id="load" checked={showLoad} onCheckedChange={setShowLoad} />
                    <Label htmlFor="load" className="text-xs font-normal">include checkpoint load ({fmtSeconds(load.end - load.start)})</Label>
                  </div>
                ) : <span>model was already in memory</span>
              }
            >
              <TimelineBar segments={segments} selected={level} onSelect={setLevel} />
              <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-xs text-muted-foreground">
                <span className="flex items-center gap-1.5"><span className="size-2.5 rounded-sm bg-primary/40" /> level step</span>
                {showLoad && load && <span className="flex items-center gap-1.5"><span className="size-2.5 rounded-sm bg-ar/30" /> checkpoint load</span>}
                <span className="flex items-center gap-1.5"><span className="size-2.5 rounded-sm bg-ok/40" /> decode to text</span>
                <span>Positions from event timestamps (`t`); durations below are the device-synchronised per-level timings from the trace.</span>
              </div>
            </Panel>

            <div className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
              <Panel title="Stages" bodyClassName="p-0">
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="text-xs text-muted-foreground">
                      <tr className="border-b border-border">
                        {['stage', 'codes', 'step', 'time', 'unique', 'H bits', 'cos in'].map((h, i) => (
                          <th key={h} className={cn('px-3 py-2 font-normal', i < 3 ? 'text-left' : 'text-right')}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="font-mono text-[0.78rem] tabular">
                      {load && (
                        <tr className="border-b border-border text-muted-foreground">
                          <td className="px-3 py-1.5">load</td><td /><td className="px-3">checkpoint</td>
                          <td className="px-3 text-right">{fmtSeconds(load.end - load.start)}</td><td /><td /><td />
                        </tr>
                      )}
                      {trace.levels.map((l) => (
                        <tr
                          key={l.level}
                          onClick={() => setLevel(l.level)}
                          className={cn('cursor-pointer border-b border-border', l.level === level ? 'bg-primary/10' : 'hover:bg-muted/40')}
                        >
                          <td className="px-3 py-1.5"><span className="inline-flex items-center gap-2"><StatusDot tone="ok" />L{l.level}</span></td>
                          <td className="px-3">{l.n_codes}</td>
                          <td className="px-3 text-muted-foreground">{l.source === 'unigram' ? 'unigram draw' : 'parallel'}</td>
                          <td className="px-3 text-right">{fmtSeconds(l.seconds)}</td>
                          <td className="px-3 text-right">{l.stats.unique}</td>
                          <td className="px-3 text-right">{l.stats.entropy_bits.toFixed(2)}</td>
                          <td className="px-3 text-right">{l.latent_in ? l.latent_in.cos_to_final.toFixed(3) : '—'}</td>
                        </tr>
                      ))}
                      {decode && (
                        <tr className="text-muted-foreground">
                          <td className="px-3 py-1.5">decode</td><td /><td className="px-3">latent → text</td>
                          <td className="px-3 text-right">{fmtSeconds(record.timing?.decode_s)}</td><td /><td /><td />
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </Panel>

              <div className="space-y-6">
                <Panel title={lv ? `Stage L${lv.level}` : 'Stage'}>
                  {lv ? (
                    <>
                      <KV k="codes predicted in one step" v={lv.n_codes} />
                      <KV k="logits" v={lv.logits_shape ? `[${lv.logits_shape.join(', ')}]` : '— (unigram draw)'} />
                      <KV k="wall time" v={fmtSeconds(lv.seconds)} />
                      <KV k="codebook unique" v={`${lv.stats.unique} / ${lv.stats.vocab_size}`} />
                      <div className="mt-3 mb-1 text-xs text-muted-foreground">Decoded after this level · sample 0 (official decode_multiscales)</div>
                      <p className="rounded-md border border-border bg-background/40 p-2.5 text-[0.82rem] leading-relaxed">
                        {lv.decoded[0]?.replace(/\s+/g, ' ').trim() || '(empty)'}
                      </p>
                      <Link
                        to={`/concepts?id=${selected}&level=${lv.level}`}
                        className="mt-3 inline-flex items-center gap-1 text-xs text-primary hover:underline"
                      >
                        Inspect codes in Concept Explorer <ArrowRight className="size-3" />
                      </Link>
                    </>
                  ) : (
                    <p className="text-sm text-muted-foreground">Select a level in the timeline or the table.</p>
                  )}
                </Panel>
                <Panel title="Time per level" meta="trace seconds">
                  <div className="h-44">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={perLevel} margin={{ left: -14, right: 4, top: 4, bottom: 0 }}>
                        <CartesianGrid stroke="var(--border)" vertical={false} />
                        <XAxis dataKey="level" tick={axis} tickLine={false} axisLine={false} interval={0} />
                        <YAxis tick={axis} tickLine={false} axisLine={false} tickFormatter={(v: number) => `${(v * 1000).toFixed(0)}`} unit=" ms" width={60} />
                        <Tooltip
                          cursor={{ fill: 'var(--muted)' }}
                          contentStyle={{ background: 'var(--popover)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }}
                          formatter={(v, _n, p) => [`${fmtSeconds(v as number)} · ${p.payload.n} codes`, p.payload.src]}
                        />
                        <Bar dataKey="seconds" fill="var(--ncm)" radius={[3, 3, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <p className="mt-2 text-xs text-muted-foreground">
                    One run on {trace.device}, batch {trace.batch_size}. Single measurements: see Benchmarks for repeated, percentiled timings.
                  </p>
                </Panel>
              </div>
            </div>
          </>
        )}
      </div>
    </>
  )
}
