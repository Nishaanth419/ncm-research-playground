import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

import { KV, Panel } from '@/components/common'
import { Badge } from '@/components/ui/badge'
import { fmtSeconds } from '@/lib/format'
import { codeColor } from '@/lib/trace'
import type { LevelTrace, Trace } from '@/lib/types'

const axis = { fill: 'var(--muted-foreground)', fontSize: 11, fontFamily: 'var(--font-mono)' }

export function LevelDetail({ trace, level, sample }: { trace: Trace; level: LevelTrace; sample: number }) {
  const s = level.stats
  const lat = level.latent_in
  const top = s.top.slice(0, 12).map((t) => ({ code: String(t.code), count: t.count, raw: t.code }))
  const row = level.codes[sample] ?? []
  return (
    <Panel
      title={<>Level {level.level} detail</>}
      meta={<span className="font-mono">{level.n_codes} codes · {level.source}</span>}
    >
      <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
        <div>
          <div className="mb-1.5 text-xs text-muted-foreground">Execution</div>
          <KV k="step" v={level.source === 'unigram' ? 'unigram draw (no transformer)' : 'one parallel forward pass'} />
          <KV k="logits shape" v={level.logits_shape ? `[${level.logits_shape.join(', ')}]` : '—'} />
          <KV k="wall time" v={fmtSeconds(level.seconds)} />
          <div className="mt-4 mb-1.5 text-xs text-muted-foreground">Codebook usage · whole batch of {trace.batch_size}</div>
          <KV k="codes emitted" v={s.n_codes.toLocaleString()} />
          <KV k="unique / codebook" v={`${s.unique} / ${s.vocab_size} (${(s.utilization * 100).toFixed(1)}%)`} />
          <KV k="entropy" v={`${s.entropy_bits.toFixed(2)} bits`} />
          <KV k="perplexity" v={s.perplexity.toFixed(1)} />
          <div className="mt-4 mb-1.5 text-xs text-muted-foreground">
            Input latent ẑ (sum of levels 0…{level.level - 1}) · first {lat?.n_samples ?? trace.latent_samples} samples
          </div>
          {lat ? (
            <>
              <KV k="‖ẑ‖ per position" v={lat.norm_mean.toFixed(3)} />
              <KV k="‖Δẑ‖ vs previous level" v={lat.delta_norm_mean.toFixed(3)} />
              <KV k="cos(ẑ, final latent)" v={lat.cos_to_final.toFixed(4)} />
            </>
          ) : (
            <p className="text-xs text-muted-foreground">Level 0 has no input latent: it is the first level.</p>
          )}
        </div>
        <div className="min-w-0">
          <div className="mb-1.5 text-xs text-muted-foreground">Most frequent codes · whole batch</div>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={top} layout="vertical" margin={{ left: 4, right: 12, top: 0, bottom: 0 }}>
                <XAxis type="number" tick={axis} axisLine={false} tickLine={false} allowDecimals={false} />
                <YAxis type="category" dataKey="code" tick={axis} axisLine={false} tickLine={false} width={44} />
                <Tooltip
                  cursor={{ fill: 'var(--muted)' }}
                  contentStyle={{ background: 'var(--popover)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }}
                  formatter={(v) => [v, 'count']}
                  labelFormatter={(l) => `code ${l}`}
                />
                <Bar dataKey="count" fill="var(--ncm)" radius={[0, 3, 3, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="mt-5">
        <div className="mb-1.5 flex items-center gap-2 text-xs text-muted-foreground">
          Code indices · sample {sample}
          <Badge variant="outline" className="font-normal">{row.length} values</Badge>
        </div>
        <div className="flex flex-wrap gap-1 font-mono text-[0.7rem] tabular">
          {row.map((c, i) => (
            <span key={i} className="rounded border border-border px-1 py-px" style={{ borderLeft: `3px solid ${codeColor(c)}` }} title={`position ${i + 1}`}>
              {c}
            </span>
          ))}
        </div>
      </div>

      <div className="mt-5">
        <div className="mb-1.5 text-xs text-muted-foreground">Decoded after level {level.level} · sample {sample}</div>
        <p className="rounded-lg border border-border bg-background/40 p-3 text-[0.86rem] leading-relaxed whitespace-pre-wrap">
          {level.decoded[sample]?.replace(/^\n+/, '') || <span className="text-muted-foreground">(empty)</span>}
        </p>
      </div>
    </Panel>
  )
}
