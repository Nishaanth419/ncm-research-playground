import { Pause, Play, RotateCcw } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { StepProgress } from '@/features/compare/StepProgress'
import { EventLog } from '@/features/playground/EventLog'
import { LevelStrip } from '@/features/playground/LevelStrip'
import type { GenEvent, ModelInfo } from '@/lib/types'
import type { LevelState } from '@/store/playground'

/**
 * Re-plays the events recorded for an experiment at their recorded offsets
 * (`t`). Nothing is recomputed: this is a playback of what was logged.
 */
export function Replay({ events, model }: { events: GenEvent[]; model?: ModelInfo }) {
  const [shown, setShown] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const timers = useRef<ReturnType<typeof setTimeout>[]>([])

  const clear = () => { timers.current.forEach(clearTimeout); timers.current = [] }
  useEffect(() => clear, [])

  const play = () => {
    clear()
    const from = shown >= events.length ? 0 : shown
    if (from === 0) setShown(0)
    const t0 = events[from]?.t ?? 0
    setPlaying(true)
    events.slice(from).forEach((e, i) => {
      timers.current.push(setTimeout(() => {
        setShown(from + i + 1)
        if (from + i + 1 === events.length) setPlaying(false)
      }, ((e.t - t0) * 1000) / speed))
    })
  }
  const pause = () => { clear(); setPlaying(false) }

  const visible = events.slice(0, shown)
  const levels: Record<number, LevelState> = {}
  let steps = 0
  for (const e of visible) {
    if (e.type === 'generation.level_started' && e.level !== undefined) levels[e.level] = { status: 'running', seconds: null }
    if (e.type === 'generation.level_completed' && e.level !== undefined) levels[e.level] = { status: 'done', seconds: e.seconds ?? null, source: e.source }
    if (e.type === 'generation.step_completed' && e.step !== undefined) steps = e.step
  }
  const done = shown === events.length

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        {playing
          ? <Button variant="outline" size="sm" onClick={pause}><Pause />Pause</Button>
          : <Button variant="outline" size="sm" onClick={play}>{done ? <RotateCcw /> : <Play />}{done ? 'Replay again' : shown ? 'Resume' : 'Replay recorded events'}</Button>}
        {[0.25, 1, 4].map((s) => (
          <button key={s} type="button" onClick={() => setSpeed(s)}
            className={`h-7 rounded-md border px-2 font-mono text-xs ${speed === s ? 'border-primary/60 text-primary' : 'border-border text-muted-foreground'}`}>
            {s}×
          </button>
        ))}
        <span className="font-mono text-xs text-muted-foreground">{shown}/{events.length} events · recorded timing</span>
      </div>
      {model?.family === 'autoregressive'
        ? <StepProgress steps={steps} max={model.output_tokens} done={done} />
        : model && model.levels.length > 0 && <LevelStrip schedule={model.levels} levels={levels} />}
      <div className="mt-4 rounded-md border border-border bg-background/40 p-2">
        <EventLog events={visible} />
      </div>
    </div>
  )
}
