import { ArrowDown } from 'lucide-react'
import type { ReactNode } from 'react'

import { fmtSeconds } from '@/lib/format'
import type { Phase, Step } from '@/lib/howitworks'
import type { Trace } from '@/lib/types'
import { cn } from '@/lib/utils'

/** The two learned parts and what each does for the level being played. */
export function Pipeline({ trace, step }: { trace: Trace; step: Step }) {
  const lv = trace.levels[step.level]
  const unigram = lv.source === 'unigram'
  const prior = step.level === 0 ? 'empty latent' : `latent with levels 0–${step.level - 1}`

  return (
    <div className="space-y-1">
      <Box active={step.phase === 'predict'} phase="predict" part="Next-concept transformer"
        title={unigram ? `Level 0: one code drawn from the unigram` : `Predict level ${step.level}: ${lv.n_codes} codes in one pass`}>
        {unigram
          ? 'TinyStories starts from a single code sampled from a stored frequency table, so this level costs no transformer pass.'
          : <>Reads the {prior} and outputs all {lv.n_codes} codes of this level at once, in parallel{lv.seconds != null && <> ({fmtSeconds(lv.seconds)} on {trace.device})</>}.</>}
      </Box>
      <Arrow />
      <Box active={step.phase === 'add'} phase="add" part="VQ-VAE quantizer"
        title={`Add level ${step.level} to the latent`}>
        Looks the codes up in level {step.level}&rsquo;s codebook ({lv.stats.vocab_size} entries), stretches them over all {trace.levels_schedule[trace.levels_schedule.length - 1]} positions and adds them to the running latent. Each level refines what the earlier ones left.
      </Box>
      <Arrow />
      <Box active={step.phase === 'decode'} phase="decode" part="VQ-VAE decoder"
        title="Decode the latent into text">
        Turns the whole latent into tokens in one step. Generation does this once at the end; this replay shows it after every level to reveal what each level added.
      </Box>
    </div>
  )
}

const PHASE_LABEL: Record<Phase, string> = { predict: '1', add: '2', decode: '3' }

function Box({ active, phase, part, title, children }: { active: boolean; phase: Phase; part: string; title: string; children: ReactNode }) {
  return (
    <div className={cn(
      'rounded-lg border p-3 transition-all duration-300',
      active ? 'border-primary/60 bg-primary/10 shadow-[0_0_0_3px] shadow-primary/10' : 'border-border bg-background/30 opacity-60',
    )}>
      <div className="mb-1 flex items-center gap-2">
        <span className={cn('flex size-5 items-center justify-center rounded-full font-mono text-[0.68rem]', active ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground')}>
          {PHASE_LABEL[phase]}
        </span>
        <span className="text-[0.68rem] tracking-wider text-muted-foreground uppercase">{part}</span>
      </div>
      <div className={cn('text-sm font-medium', active && 'text-primary')}>{title}</div>
      <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{children}</p>
    </div>
  )
}

function Arrow() {
  return <div className="flex justify-center text-muted-foreground/60"><ArrowDown className="size-4" /></div>
}
