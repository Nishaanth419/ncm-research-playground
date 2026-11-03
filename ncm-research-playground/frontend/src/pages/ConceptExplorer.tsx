import { ChevronLeft, ChevronRight, FlaskConical, Info } from 'lucide-react'
import { useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'

import { EmptyState, PageHeader, Panel } from '@/components/common'
import { ExperimentPicker } from '@/components/ExperimentPicker'
import { Button } from '@/components/ui/button'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { HierarchyGrid, type ColorMode } from '@/features/concepts/HierarchyGrid'
import { LevelDetail } from '@/features/concepts/LevelDetail'
import { LevelSeries } from '@/features/concepts/LevelSeries'
import { TextLadder } from '@/features/concepts/TextLadder'
import { useExperimentTrace } from '@/hooks/useExperimentTrace'
import type { Trace } from '@/lib/types'

export function ConceptExplorer() {
  const { candidates, selected, select, trace, error, loading } = useExperimentTrace()
  const [params] = useSearchParams()
  const requested = params.has('level') ? Number(params.get('level')) : Number.NaN

  return (
    <>
      <PageHeader
        title="Concept Explorer"
        description="The hierarchy of discrete codes NCM actually generated: one row per level, each predicted in a single parallel step, coarse to fine."
        actions={<ExperimentPicker candidates={candidates} selected={selected} onSelect={select} />}
      />
      <div className="space-y-6 p-4 sm:p-6 lg:p-8">
        {error && <div className="rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">{error}</div>}
        {candidates?.length === 0 && (
          <EmptyState icon={FlaskConical} title="No traced experiments yet">
            Run something in the <Link to="/playground" className="text-primary hover:underline">Playground</Link>; its trace appears here.
          </EmptyState>
        )}
        {trace && (
          <ConceptBody key={selected ?? ''} trace={trace}
            initialLevel={requested >= 0 && requested < trace.levels.length ? requested : trace.levels.length - 1} />
        )}
        {loading && !trace && <div className="text-sm text-muted-foreground">Loading trace…</div>}
      </div>
    </>
  )
}

function ConceptBody({ trace, initialLevel }: { trace: Trace; initialLevel: number }) {
  // keyed by experiment id: state starts fresh for every experiment
  const [level, setLevel] = useState(initialLevel)
  const [sample, setSample] = useState(0)
  const [mode, setMode] = useState<ColorMode>('id')
  const nDetail = Math.min(trace.batch_size, trace.max_detail_samples)
  const lv = trace.levels[level]
  // only the XSum config enables pick_tokens penalties (frequency/adjacency)
  const penalised = trace.task === 'xsum'
  return (
    <>
      <Panel
        title="Code hierarchy"
        meta={
          <div className="flex items-center gap-3">
            <span className="hidden font-mono sm:inline">{trace.levels.length} levels · {trace.levels_schedule.reduce((a, b) => a + b, 0)} codes / sample</span>
          </div>
        }
      >
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-1.5">
            <span className="text-xs text-muted-foreground">sample</span>
            <Button variant="outline" size="icon-sm" disabled={sample === 0} onClick={() => setSample((s) => s - 1)}><ChevronLeft /></Button>
            <span className="w-12 text-center font-mono text-xs tabular">{sample + 1} / {nDetail}</span>
            <Button variant="outline" size="icon-sm" disabled={sample >= nDetail - 1} onClick={() => setSample((s) => s + 1)}><ChevronRight /></Button>
            {trace.batch_size > nDetail && <span className="text-xs text-muted-foreground">(codes kept for first {nDetail} of {trace.batch_size})</span>}
          </div>
          <Tabs value={mode} onValueChange={(v) => setMode(v as ColorMode)}>
            <TabsList>
              <TabsTrigger value="id">colour = code id</TabsTrigger>
              <TabsTrigger value="frequency">colour = frequency</TabsTrigger>
            </TabsList>
          </Tabs>
        </div>
        <HierarchyGrid trace={trace} sample={sample} selected={level} onSelect={setLevel} mode={mode} />
        <p className="mt-3 text-xs text-muted-foreground">
          {mode === 'id'
            ? 'Categorical colour: the same code id has the same colour within the view; hue says nothing about how similar two codes are.'
            : `Opacity = how often the code occurs at that level across the ${nDetail} stored sample(s).`}
          {' '}Each row spans the whole sequence; level k's codes are upsampled and added into the latent before level k+1 is predicted.
        </p>
      </Panel>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(0,0.8fr)]">
        <LevelDetail trace={trace} level={lv} sample={sample} />
        <div className="space-y-6">
          <Panel title="Coarse-to-fine decodes" meta={`sample ${sample}`}>
            <TextLadder trace={trace} sample={sample} selected={level} onSelect={setLevel} />
            <p className="mt-3 flex gap-2 text-xs text-muted-foreground">
              <Info className="mt-0.5 size-3.5 shrink-0" />
              <span>
                L-rows: official <code className="font-mono">decode_multiscales</code>, i.e. the decoder's argmax on the cumulative latent after that level (real model outputs, not truncations of the final text).
                {' '}<span className="text-ok">out</span>: the final output{penalised ? ', decoded with the official repetition penalties (pick_tokens), so it can differ from the last L-row' : ''}.
              </span>
            </p>
          </Panel>
          <Panel title="Per-level statistics">
            <LevelSeries trace={trace} selected={level} />
            <p className="mt-3 text-xs text-muted-foreground">
              Codebook statistics over the whole batch; latent cosine over the first {trace.latent_samples} sample(s). Level 0 has no input latent.
            </p>
          </Panel>
        </div>
      </div>
    </>
  )
}
