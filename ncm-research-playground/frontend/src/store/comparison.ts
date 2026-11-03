import { create } from 'zustand'

import { api, runExperiment } from '@/lib/api'
import type { ExperimentRecord, GenEvent, SamplingParams } from '@/lib/types'
import type { LevelState } from '@/store/playground'

export type Side = 'ncm' | 'ar'
export interface SideState {
  status: 'idle' | 'queued' | 'loading-model' | 'running' | 'scoring' | 'done' | 'failed'
  levels: Record<number, LevelState>
  steps: number
  record: ExperimentRecord | null
  error: string | null
}
export type Phase = 'idle' | 'running' | 'scoring' | 'done' | 'failed'

const emptySide = (): SideState => ({ status: 'idle', levels: {}, steps: 0, record: null, error: null })

interface ComparisonState {
  models: Record<Side, string>
  numSamples: number
  seed: number
  sampling: Pick<SamplingParams, 'temperature' | 'top_k' | 'top_p'>
  phase: Phase
  tag: string | null
  sides: Record<Side, SideState>
  set: (patch: Partial<Pick<ComparisonState, 'numSamples' | 'seed' | 'sampling' | 'models'>>) => void
  run: () => Promise<void>
}

export const useComparison = create<ComparisonState>((set, get) => {
  const patchSide = (side: Side, patch: Partial<SideState> | ((s: SideState) => Partial<SideState>)) =>
    set((st) => ({ sides: { ...st.sides, [side]: { ...st.sides[side], ...(typeof patch === 'function' ? patch(st.sides[side]) : patch) } } }))

  const onEvent = (side: Side) => (e: GenEvent) =>
    patchSide(side, (s) => {
      if (e.type === 'generation.started') return { status: 'running' }
      if (e.type === 'model.loading') return { status: 'loading-model' }
      if (e.type === 'model.loaded') return { status: 'running' }
      if (e.type === 'generation.level_started' && e.level !== undefined)
        return { levels: { ...s.levels, [e.level]: { status: 'running', seconds: null } } }
      if (e.type === 'generation.level_completed' && e.level !== undefined)
        return { levels: { ...s.levels, [e.level]: { status: 'done', seconds: e.seconds ?? null, source: e.source } } }
      if (e.type === 'generation.step_completed' && e.step !== undefined) return { steps: e.step }
      return {}
    })

  return {
    models: { ncm: 'ncm-tinystories', ar: 'ar-tinystories-33m' },
    numSamples: 4,
    seed: 0,
    // shared by both sides; equal to both checkpoints' defaults (pure sampling)
    sampling: { temperature: 1, top_k: 0, top_p: 1 },
    phase: 'idle',
    tag: null,
    sides: { ncm: emptySide(), ar: emptySide() },
    set: (patch) => set(patch),

    run: async () => {
      const { models, numSamples, seed, sampling } = get()
      const tag = `cmp-${Date.now().toString(36)}`
      set({ phase: 'running', tag, sides: { ncm: { ...emptySide(), status: 'queued' }, ar: { ...emptySide(), status: 'queued' } } })
      // sequential on purpose: the backend runs one generation at a time, so
      // neither model's timing is measured while the other competes for the CPU
      for (const side of ['ncm', 'ar'] as Side[]) {
        try {
          const record = await runExperiment(
            { model: models[side], num_samples: numSamples, seed, sampling, include_trace: side === 'ncm', tag, warmup: true },
            onEvent(side))
          patchSide(side, { record, status: 'done' })
        } catch (e) {
          patchSide(side, { status: 'failed', error: (e as Error).message })
          set({ phase: 'failed' })
          return
        }
      }
      set({ phase: 'scoring' })
      for (const side of ['ncm', 'ar'] as Side[]) {
        const rec = get().sides[side].record!
        patchSide(side, { status: 'scoring' })
        try {
          await api.quality(rec.id)
          patchSide(side, { record: await api.experiment(rec.id), status: 'done' })
        } catch (e) {
          patchSide(side, { status: 'done', error: `quality scoring failed: ${(e as Error).message}` })
        }
      }
      set({ phase: 'done' })
    },
  }
})
