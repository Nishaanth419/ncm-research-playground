import { useMemo } from 'react'

import { codeColor, codeCounts } from '@/lib/trace'
import type { Trace } from '@/lib/types'
import { cn } from '@/lib/utils'

export type ColorMode = 'id' | 'frequency'

/**
 * Every level is a full-width row: each level describes the whole sequence,
 * coarse levels with few codes, fine levels with many. Row k = the n_k codes
 * predicted in one parallel step.
 */
export function HierarchyGrid({ trace, sample, selected, onSelect, mode }: {
  trace: Trace
  sample: number
  selected: number
  onSelect: (level: number) => void
  mode: ColorMode
}) {
  const counts = useMemo(() => trace.levels.map((lv) => codeCounts(lv.codes)), [trace])
  return (
    <div className="space-y-[3px]">
      {trace.levels.map((lv, k) => {
        const row = lv.codes[sample] ?? []
        const max = Math.max(1, ...counts[k].values())
        return (
          <button
            key={k}
            type="button"
            onClick={() => onSelect(k)}
            className={cn(
              'group flex w-full items-center gap-2 rounded-[4px] py-[2px] pr-1 text-left outline-none',
              k === selected ? 'bg-primary/10 ring-1 ring-primary/50' : 'hover:bg-muted/50',
            )}
          >
            <span className={cn('w-[4.5rem] shrink-0 pl-1.5 font-mono text-[0.68rem] whitespace-nowrap tabular', k === selected ? 'text-primary' : 'text-muted-foreground')}>
              L{k} · {lv.n_codes}
            </span>
            <span className="flex h-3.5 min-w-0 flex-1 gap-px">
              {row.map((c, i) => (
                <span
                  key={i}
                  title={`L${k} · position ${i + 1}/${lv.n_codes} · code ${c} · ×${counts[k].get(c)} in ${lv.codes.length} sample(s)`}
                  className="min-w-0 flex-1 rounded-[1px]"
                  style={mode === 'id'
                    ? { background: codeColor(c) }
                    : { background: `color-mix(in oklch, var(--ncm) ${Math.round(15 + 85 * ((counts[k].get(c) ?? 1) / max))}%, transparent)` }}
                />
              ))}
            </span>
          </button>
        )
      })}
    </div>
  )
}
