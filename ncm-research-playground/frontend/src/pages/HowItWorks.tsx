import { ChevronLeft, ChevronRight, Loader2, Sparkles } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'

import { EmptyState, PageHeader, Panel } from '@/components/common'
import { ExperimentPicker } from '@/components/ExperimentPicker'
import { Button } from '@/components/ui/button'
import { CoarseToFine } from '@/features/howitworks/CoarseToFine'
import { PassRace, type MeasuredSpeed } from '@/features/howitworks/PassRace'
import { Pipeline } from '@/features/howitworks/Pipeline'
import { PlayerControls } from '@/features/howitworks/PlayerControls'
import { useExperimentTrace } from '@/hooks/useExperimentTrace'
import { usePlayer } from '@/hooks/usePlayer'
import { api, runExperiment } from '@/lib/api'
import { playbackSteps } from '@/lib/howitworks'
import type { ExperimentRecord, Trace } from '@/lib/types'

const NCM_MODEL = 'ncm-tinystories'
// the explainer and the measured timings are about the unconditional TinyStories model
const isTinyStories = (r: ExperimentRecord) => r.model === NCM_MODEL
const AR_MODEL = 'ar-tinystories-33m'

export function HowItWorks() {
  const { candidates, selected, select, reload, trace, error, loading } = useExperimentTrace(isTinyStories)
  const measured = useMeasuredSpeed()
  const [running, setRunning] = useState(false)
  const [runError, setRunError] = useState<string | null>(null)

  const generate = async () => {
    setRunning(true)
    setRunError(null)
    try {
      const rec = await runExperiment({ model: NCM_MODEL, num_samples: 1, seed: 0, sampling: {}, include_trace: true, tag: 'how-it-works' }, () => {})
      reload()
      select(rec.id)
    } catch (e) {
      setRunError((e as Error).message)
    } finally {
      setRunning(false)
    }
  }

  return (
    <>
      <PageHeader
        title="How NCM works"
        description="Next-Concept Modeling writes a whole passage coarse to fine: a few codes for the gist first, then more and more detail, each level predicted in one parallel step. Played back from a real recorded run."
        actions={candidates && candidates.length > 0 && <ExperimentPicker candidates={candidates} selected={selected} onSelect={select} />}
      />
      <div className="space-y-6 p-4 sm:p-6 lg:p-8">
        <Intro />
        {(error || runError) && <div className="rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">{runError ?? error}</div>}
        {candidates?.length === 0 && (
          <EmptyState icon={Sparkles} title="No recorded NCM TinyStories run to play back yet">
            <p>The animation replays a real run. Generate one (about a second on CPU, plus loading the checkpoint the first time).</p>
            <Button className="mt-3" onClick={() => void generate()} disabled={running}>
              {running ? <><Loader2 className="animate-spin" />Generating…</> : <><Sparkles />Generate a TinyStories run</>}
            </Button>
          </EmptyState>
        )}
        {loading && !trace && <div className="text-sm text-muted-foreground">Loading trace…</div>}
        {trace && <Scenes key={selected ?? ''} id={selected ?? ''} trace={trace} measured={measured} />}
      </div>
    </>
  )
}

function Intro() {
  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
      <div className="rounded-xl border border-border bg-card p-4">
        <div className="mb-1 text-sm font-medium text-ar">A traditional language model</div>
        <p className="text-sm text-muted-foreground">Writes left to right, one token per forward pass. A 128-token story takes up to 128 passes, and each token is fixed before the next is chosen.</p>
      </div>
      <div className="rounded-xl border border-border bg-card p-4">
        <div className="mb-1 text-sm font-medium text-ncm">Next-Concept Modeling</div>
        <p className="text-sm text-muted-foreground">
          A VQ-VAE compresses text into a stack of levels of discrete codes, from 1 code for the whole passage up to 128. A transformer predicts the next level from the ones before it, all codes of a level at once. TinyStories needs 11 passes.
        </p>
      </div>
    </div>
  )
}

function Scenes({ id, trace, measured }: { id: string; trace: Trace; measured: MeasuredSpeed | null }) {
  const nDetail = Math.min(trace.batch_size, trace.max_detail_samples)
  const [sample, setSample] = useState(0)
  const steps = useMemo(() => playbackSteps(trace.levels.length), [trace])
  const [speed, setSpeed] = useState(1)
  const player = usePlayer(steps.length, 900 / speed)
  const step = steps[player.index]

  return (
    <>
      <Panel title="1 · One token at a time vs. one level at a time" meta={<span className="font-mono">{trace.task} · {trace.levels.length} levels</span>}>
        <PassRace trace={trace} sample={sample} measured={measured} />
      </Panel>

      <Panel title="2 · Coarse to fine, replayed" meta={
        nDetail > 1 && (
          <span className="flex items-center gap-1.5">
            sample
            <Button variant="outline" size="icon-xs" aria-label="previous sample" disabled={sample === 0} onClick={() => setSample((s) => s - 1)}><ChevronLeft /></Button>
            <span className="w-10 text-center font-mono tabular">{sample + 1} / {nDetail}</span>
            <Button variant="outline" size="icon-xs" aria-label="next sample" disabled={sample >= nDetail - 1} onClick={() => setSample((s) => s + 1)}><ChevronRight /></Button>
          </span>
        )
      }>
        <div className="mb-5">
          <PlayerControls player={player} length={steps.length} speed={speed} onSpeed={setSpeed}
            label={<>level <span className="text-foreground">{step.level}</span> / {trace.levels.length - 1} · step {step.phase}</>} />
        </div>
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
          <CoarseToFine trace={trace} sample={sample} step={step} />
          <Pipeline trace={trace} step={step} />
        </div>
        <p className="mt-4 text-xs text-muted-foreground">
          Codes, per-level times and texts are this run&rsquo;s recorded values; colour = code id (same id, same colour; hue says nothing about similarity).
          The per-level texts are the official <code className="font-mono">decode_multiscales</code> output: the decoder&rsquo;s reading of the latent after that level, not cut-offs of the final story.
          Explore the same run in the <Link to={`/concepts?id=${id}`} className="text-primary hover:underline">Concept Explorer</Link>.
        </p>
      </Panel>
    </>
  )
}

/** Batch-1 medians from the most recent completed benchmark that measured both TinyStories models. */
function useMeasuredSpeed(): MeasuredSpeed | null {
  const [m, setM] = useState<MeasuredSpeed | null>(null)
  useEffect(() => {
    let live = true
    api.benchmarks(20).then(async (rows) => {
      for (const b of rows.filter((r) => r.status === 'completed')) {
        const full = b.summary ? b : await api.benchmark(b.id)
        const pick = (model: string) => {
          const g = full.summary?.find((x) => x.model === model && x.batch_size === 1)
          return g && g.total_s.median != null ? { totalS: g.total_s.median, passes: g.passes } : null
        }
        const ncm = pick(NCM_MODEL)
        const ar = pick(AR_MODEL)
        if (ncm || ar) { if (live) setM({ benchmarkId: b.id, ncm, ar }); return }
      }
    }, () => {})
    return () => { live = false }
  }, [])
  return m
}
