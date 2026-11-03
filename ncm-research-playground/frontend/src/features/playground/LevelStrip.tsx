import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { fmtSeconds } from '@/lib/format'
import { cn } from '@/lib/utils'
import type { LevelState } from '@/store/playground'

/**
 * One cell per level, width proportional to the level's code count.
 * Levels run left to right (sequential); every cell is a single parallel step.
 */
export function LevelStrip({ schedule, levels }: { schedule: number[]; levels: Record<number, LevelState> }) {
  const done = Object.values(levels).filter((l) => l.status === 'done').length
  return (
    <div>
      <div className="flex h-12 items-stretch gap-[3px]">
        {schedule.map((n, i) => {
          const st = levels[i]?.status ?? 'pending'
          return (
            <Tooltip key={i}>
              <TooltipTrigger
                render={
                  <div
                    style={{ flexGrow: n, flexBasis: 0 }}
                    className={cn(
                      'relative min-w-[10px] overflow-hidden rounded-[3px] border transition-colors duration-300',
                      st === 'done' && 'border-primary/50 bg-primary/25',
                      st === 'running' && 'border-primary bg-primary/10',
                      st === 'pending' && 'border-border bg-muted/40',
                    )}
                  >
                    {st === 'running' && <div className="absolute inset-0 animate-pulse bg-primary/20" />}
                    <span className="absolute bottom-0.5 left-1 font-mono text-[0.62rem] text-muted-foreground tabular">{i}</span>
                  </div>
                }
              />
              <TooltipContent>
                <div className="font-mono text-xs">
                  L{i} · {n} codes · {levels[i]?.source ?? (i === 0 ? 'level 0' : 'transformer')}
                  {levels[i]?.seconds != null && <> · {fmtSeconds(levels[i].seconds)}</>}
                </div>
              </TooltipContent>
            </Tooltip>
          )
        })}
      </div>
      <div className="mt-2 flex justify-between text-[0.7rem] text-muted-foreground">
        <span>coarse · {schedule[0]} code{schedule[0] === 1 ? '' : 's'}</span>
        <span className="font-mono tabular">{done}/{schedule.length} levels</span>
        <span>fine · {schedule[schedule.length - 1]} codes</span>
      </div>
    </div>
  )
}
