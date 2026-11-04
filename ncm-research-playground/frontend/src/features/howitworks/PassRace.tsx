import { Check } from 'lucide-react'
import { useState } from 'react'

import { usePlayer } from '@/hooks/usePlayer'
import { ncmPasses, raceState } from '@/lib/howitworks'
import { codeColor } from '@/lib/trace'
import type { Trace } from '@/lib/types'
import { cn } from '@/lib/utils'

import { PlayerControls } from './PlayerControls'

export interface MeasuredSpeed {
  benchmarkId: string
  ncm: { totalS: number; passes: number } | null
  ar: { totalS: number; passes: number } | null
}

/**
 * The same 128-position canvas filled two ways, one forward pass per tick.
 * Autoregressive: one more token per pass. NCM: one more level per pass, each
 * level covering the whole canvas with more, finer codes (this run's real codes).
 */
export function PassRace({ trace, sample, measured }: { trace: Trace; sample: number; measured: MeasuredSpeed | null }) {
  const length = trace.levels_schedule[trace.levels_schedule.length - 1]
  const [speed, setSpeed] = useState(1)
  const player = usePlayer(length + 1, 90 / speed)
  const passes = player.index
  const s = raceState(passes, trace)
  const top = s.ncmLevels - 1
  const codes = top >= 0 ? trace.levels[top].codes[sample] ?? [] : []

  return (
    <div className="space-y-5">
      <PlayerControls player={player} length={length + 1} speed={speed} onSpeed={setSpeed}
        label={<>forward pass <span className="text-foreground">{passes}</span> / {length}</>} />

      <Lane
        title="Autoregressive (one token per pass)"
        tone="ar"
        status={s.arDone ? `done after ${length} passes` : `${s.arTokens} / ${length} tokens`}
        done={s.arDone}
      >
        {Array.from({ length }, (_, i) => (
          <span key={i} className={cn('min-w-0 flex-1 border-r border-background/60 transition-colors duration-150', i < s.arTokens ? 'bg-ar' : 'bg-muted/60')} />
        ))}
      </Lane>

      <Lane
        title="NCM (one level per pass, whole canvas each time)"
        tone="ncm"
        status={s.ncmDone ? `done after ${ncmPasses(trace)} passes` : top < 0 ? 'nothing yet' : `level ${top} · ${trace.levels[top].n_codes} code${trace.levels[top].n_codes === 1 ? '' : 's'}`}
        done={s.ncmDone}
      >
        {codes.length === 0
          ? <span className="flex-1 rounded-[1px] bg-muted/60" />
          : codes.map((c, i) => (
            <span key={`${top}-${i}`} title={`level ${top} · code ${c}`}
              className="min-w-0 flex-1 border-r border-background/60 motion-safe:animate-[ncm-in_260ms_ease-out]" style={{ background: codeColor(c) }} />
          ))}
      </Lane>

      <div className="space-y-1.5 text-xs text-muted-foreground">
        <p>
          The NCM row is real: the codes this run produced at the newest level, each spread over its share of the canvas.
          The autoregressive row is schematic: it only shows the pass count (the baseline may also stop early at end-of-text).
          One tick is one forward pass, slowed down; passes of the two models do not cost the same time.
        </p>
        {measured && (measured.ncm || measured.ar) && (
          <p>
            Measured on this machine (benchmark <span className="font-mono">{measured.benchmarkId.slice(0, 8)}</span>, batch 1, median):
            {measured.ncm && <> <span className="text-ncm">NCM {Math.round(measured.ncm.totalS * 1000)} ms</span> for {measured.ncm.passes} passes</>}
            {measured.ncm && measured.ar && ','}
            {measured.ar && <> <span className="text-ar">baseline {Math.round(measured.ar.totalS * 1000)} ms</span> for up to {measured.ar.passes} passes</>}.
          </p>
        )}
      </div>
    </div>
  )
}

function Lane({ title, tone, status, done, children }: {
  title: string
  tone: 'ar' | 'ncm'
  status: string
  done: boolean
  children: React.ReactNode
}) {
  return (
    <div>
      <div className="mb-1.5 flex flex-wrap items-baseline justify-between gap-2 text-sm">
        <span className={tone === 'ar' ? 'text-ar' : 'text-ncm'}>{title}</span>
        <span className={cn('flex items-center gap-1 font-mono text-xs tabular', done ? 'text-ok' : 'text-muted-foreground')}>
          {done && <Check className="size-3.5" />}{status}
        </span>
      </div>
      <div className="flex h-7 overflow-hidden rounded-md border border-border bg-background/40 p-1">{children}</div>
    </div>
  )
}
