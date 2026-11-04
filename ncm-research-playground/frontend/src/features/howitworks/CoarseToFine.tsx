import { fmtSeconds } from '@/lib/format'
import { decodedLevel, visibleLevels, type Step } from '@/lib/howitworks'
import { codeColor } from '@/lib/trace'
import type { Trace } from '@/lib/types'
import { cn } from '@/lib/utils'

/**
 * The run's code hierarchy revealed level by level, with the text the decoder
 * reads from the latent after each level (official decode_multiscales output).
 */
export function CoarseToFine({ trace, sample, step }: { trace: Trace; sample: number; step: Step }) {
  const shown = visibleLevels(step)
  const dl = decodedLevel(step)
  const text = dl >= 0 ? (trace.levels[dl].decoded[sample] ?? '').replace(/^\s+/, '') : ''
  const isLast = dl === trace.levels.length - 1
  const final = (trace.final_texts[sample] ?? '').replace(/^\s+/, '')

  return (
    <div className="space-y-4">
      <div className="space-y-[3px]">
        {trace.levels.map((lv, k) => {
          const on = k < shown
          const current = k === step.level
          const row = lv.codes[sample] ?? []
          return (
            <div key={k} className={cn('flex items-center gap-2 rounded-[4px] py-[2px] pr-1 transition-colors', current && 'bg-primary/10 ring-1 ring-primary/50')}>
              <span className={cn('w-[5.5rem] shrink-0 pl-1.5 font-mono text-[0.68rem] whitespace-nowrap tabular', current ? 'text-primary' : on ? 'text-muted-foreground' : 'text-muted-foreground/40')}>
                L{k} · {lv.n_codes}
              </span>
              {/* 1px gaps only while they fit; dense rows would squeeze cells to zero width */}
              <span className={cn('flex h-3.5 min-w-0 flex-1', lv.n_codes <= 48 && 'gap-px')}>
                {on
                  ? row.map((c, i) => (
                    <span key={i} className="min-w-0 flex-1 rounded-[1px] motion-safe:animate-[ncm-in_320ms_ease-out]"
                      style={{ background: codeColor(c), animationDelay: current && step.phase === 'predict' ? `${Math.min(i, 40) * 4}ms` : undefined, animationFillMode: 'both' }} />
                  ))
                  : <span className="flex-1 rounded-[1px] border border-dashed border-border/70" />}
              </span>
              <span className="hidden w-14 shrink-0 text-right font-mono text-[0.68rem] text-muted-foreground tabular sm:inline">
                {on ? (lv.source === 'unigram' ? 'unigram' : fmtSeconds(lv.seconds)) : ''}
              </span>
            </div>
          )
        })}
      </div>

      <div className="rounded-lg border border-border bg-background/40 p-3">
        <div className="mb-1.5 flex flex-wrap items-baseline justify-between gap-2 font-mono text-[0.68rem] text-muted-foreground">
          <span>{dl < 0 ? 'decoded text' : `decoded after level ${dl} (levels 0–${dl} in the latent)`}</span>
          {dl >= 0 && <span>{trace.levels.slice(0, dl + 1).reduce((a, l) => a + l.n_codes, 0)} codes so far</span>}
        </div>
        <p key={dl} className="min-h-[5.5rem] text-[0.86rem] leading-relaxed whitespace-pre-wrap motion-safe:animate-[ncm-fade_300ms_ease-out]">
          {dl < 0 ? <span className="text-muted-foreground">Nothing to decode yet: the first level has not been added to the latent.</span>
            : text || <span className="text-muted-foreground">(empty)</span>}
        </p>
        {isLast && final && final !== text && (
          <p className="mt-2 border-t border-border pt-2 text-[0.8rem] text-muted-foreground"><span className="text-ok">final output</span> (official pick_tokens): {final}</p>
        )}
      </div>
    </div>
  )
}
