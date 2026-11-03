import type { Trace } from '@/lib/types'
import { cn } from '@/lib/utils'

/** Official per-level decodes for one sample, then the final output. */
export function TextLadder({ trace, sample, selected, onSelect }: {
  trace: Trace
  sample: number
  selected: number
  onSelect: (level: number) => void
}) {
  return (
    <div className="space-y-1">
      {trace.levels.map((lv) => (
        <button
          key={lv.level}
          type="button"
          onClick={() => onSelect(lv.level)}
          className={cn(
            'flex w-full gap-3 rounded-md px-2 py-1.5 text-left text-[0.8rem] leading-snug',
            lv.level === selected ? 'bg-primary/10 text-foreground' : 'text-muted-foreground hover:bg-muted/50 hover:text-foreground',
          )}
        >
          <span className={cn('w-8 shrink-0 font-mono text-[0.68rem] tabular', lv.level === selected && 'text-primary')}>L{lv.level}</span>
          <span className="min-w-0 truncate">{lv.decoded[sample]?.replace(/\s+/g, ' ').trim() || '(empty)'}</span>
        </button>
      ))}
      <div className="mt-2 flex gap-3 rounded-md border border-border px-2 py-2 text-[0.8rem] leading-snug">
        <span className="w-8 shrink-0 font-mono text-[0.68rem] text-ok">out</span>
        <span className="min-w-0">{trace.final_texts[sample]?.replace(/\s+/g, ' ').trim()}</span>
      </div>
    </div>
  )
}
