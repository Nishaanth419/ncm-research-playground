import { ChevronLeft, ChevronRight, Pause, Play, RotateCcw } from 'lucide-react'
import type { ReactNode } from 'react'

import { Button } from '@/components/ui/button'
import { Slider } from '@/components/ui/slider'
import type { usePlayer } from '@/hooks/usePlayer'

const SPEEDS = [0.5, 1, 2, 4]

export function PlayerControls({ player, length, speed, onSpeed, label }: {
  player: ReturnType<typeof usePlayer>
  length: number
  speed: number
  onSpeed: (s: number) => void
  label: ReactNode
}) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
      <div className="flex items-center gap-1.5">
        <Button size="sm" onClick={player.toggle} className="w-24">
          {player.playing ? <><Pause />Pause</> : player.atEnd && player.index > 0 ? <><RotateCcw />Replay</> : <><Play />Play</>}
        </Button>
        <Button variant="outline" size="icon-sm" aria-label="previous step" disabled={player.index === 0} onClick={() => player.seek(player.index - 1)}><ChevronLeft /></Button>
        <Button variant="outline" size="icon-sm" aria-label="next step" disabled={player.atEnd} onClick={() => player.seek(player.index + 1)}><ChevronRight /></Button>
      </div>
      <div className="min-w-40 flex-1">
        <Slider value={[player.index]} min={0} max={Math.max(0, length - 1)} step={1}
          onValueChange={(x) => player.seek(Array.isArray(x) ? x[0] : (x as number))} />
      </div>
      <div className="flex items-center gap-1">
        {SPEEDS.map((s) => (
          <Button key={s} variant={s === speed ? 'secondary' : 'ghost'} size="xs" className="font-mono" onClick={() => onSpeed(s)}>{s}×</Button>
        ))}
      </div>
      <div className="w-full font-mono text-xs whitespace-nowrap text-muted-foreground tabular sm:w-52 sm:text-right">{label}</div>
    </div>
  )
}
