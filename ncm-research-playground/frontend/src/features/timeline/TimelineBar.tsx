import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { fmtSeconds } from '@/lib/format'
import type { Segment } from '@/lib/trace'
import { cn } from '@/lib/utils'

function ticks(max: number): number[] {
  const raw = max / 5
  const pow = 10 ** Math.floor(Math.log10(raw || 1))
  const step = [1, 2, 5, 10].map((m) => m * pow).find((s) => s >= raw) ?? raw
  return Array.from({ length: Math.floor(max / step) + 1 }, (_, i) => i * step)
}

/** Recorded execution on one time axis (event timestamps, seconds since start). */
export function TimelineBar({ segments, selected, onSelect }: {
  segments: Segment[]
  selected: number | null
  onSelect: (level: number) => void
}) {
  if (!segments.length) return null
  const t0 = Math.min(...segments.map((s) => s.start))
  const t1 = Math.max(...segments.map((s) => s.end))
  const span = Math.max(t1 - t0, 1e-6)
  const pct = (t: number) => `${((t - t0) / span) * 100}%`
  return (
    <div>
      <div className="relative h-14 rounded-md border border-border bg-background/40">
        {segments.map((s, i) => {
          const w = ((s.end - s.start) / span) * 100
          const isLevel = s.kind === 'level'
          return (
            <Tooltip key={i}>
              <TooltipTrigger
                render={
                  <button
                    type="button"
                    disabled={!isLevel}
                    onClick={() => isLevel && s.level !== undefined && onSelect(s.level)}
                    style={{ left: pct(s.start), width: w > 0 ? `max(${w}%, 2px)` : '2px' }}
                    className={cn(
                      'absolute top-2 bottom-2 rounded-[3px] border transition-colors',
                      s.kind === 'load' && 'border-ar/40 bg-ar/20',
                      s.kind === 'decode' && 'border-ok/50 bg-ok/25',
                      isLevel && (s.level === selected ? 'border-primary bg-primary/60' : 'border-primary/50 bg-primary/25 hover:bg-primary/40'),
                      s.incomplete && 'border-dashed border-destructive bg-destructive/20',
                    )}
                  />
                }
              />
              <TooltipContent>
                <span className="font-mono text-xs">
                  {s.kind === 'level' ? `L${s.level} · ${s.n_codes} codes · ${s.source ?? ''}` : s.kind === 'load' ? 'checkpoint load' : 'decode (latent → text)'}
                  {' · '}{fmtSeconds(s.end - s.start)}
                </span>
              </TooltipContent>
            </Tooltip>
          )
        })}
      </div>
      <div className="relative mt-1 h-4 font-mono text-[0.65rem] text-muted-foreground tabular">
        {ticks(span).map((t) => (
          <span key={t} className="absolute -translate-x-1/2" style={{ left: `${(t / span) * 100}%` }}>
            {fmtSeconds(t)}
          </span>
        ))}
      </div>
    </div>
  )
}
