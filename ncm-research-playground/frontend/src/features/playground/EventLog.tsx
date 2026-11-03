import { useEffect, useRef } from 'react'

import { fmtSeconds } from '@/lib/format'
import type { GenEvent } from '@/lib/types'
import { cn } from '@/lib/utils'

function describe(e: GenEvent): string {
  switch (e.type) {
    case 'generation.started': return `model=${e.model} samples=${e.num_samples} seed=${e.seed}`
    case 'model.loading': return `loading ${e.model} on ${e.device}`
    case 'model.loaded': return `loaded in ${fmtSeconds(e.seconds)}`
    case 'generation.level_started': return `L${e.level} · ${e.n_codes} codes`
    case 'generation.level_completed': return `L${e.level} · ${e.n_codes} codes · ${e.source}${e.seconds != null ? ` · ${fmtSeconds(e.seconds)}` : ''}`
    case 'generation.completed': return `generate ${fmtSeconds(e.timing?.generate_s)} · decode ${fmtSeconds(e.timing?.decode_s)}`
    case 'generation.failed': return e.error ?? ''
    case 'generation.step_completed': return `step ${e.step} · ${e.n_tokens} token(s)`
  }
}

/** Raw stream of execution events exactly as received over the WebSocket. */
export function EventLog({ events }: { events: GenEvent[] }) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    ref.current?.scrollTo({ top: ref.current.scrollHeight })
  }, [events.length])
  return (
    <div ref={ref} className="h-56 overflow-y-auto font-mono text-[0.72rem] leading-relaxed">
      {events.length === 0 && <div className="text-muted-foreground">No events yet.</div>}
      {events.map((e) => (
        <div key={e.seq} className="flex gap-3 whitespace-nowrap">
          <span className="w-14 shrink-0 text-right text-muted-foreground tabular">{e.t.toFixed(3)}</span>
          <span className={cn(
            'w-44 shrink-0',
            e.type === 'generation.failed' ? 'text-destructive' : e.type === 'generation.completed' ? 'text-ok' : e.type.startsWith('model') ? 'text-ar' : 'text-foreground/80',
          )}>{e.type}</span>
          <span className="truncate text-muted-foreground">{describe(e)}</span>
        </div>
      ))}
    </div>
  )
}
