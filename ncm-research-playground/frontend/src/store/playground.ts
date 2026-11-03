import { create } from 'zustand'

import { api, streamExperiment } from '@/lib/api'
import type { ExperimentRecord, GenEvent, GenerateRequest, SamplingParams, Trace } from '@/lib/types'

export type LevelState = { status: 'pending' | 'running' | 'done'; seconds: number | null; source?: string }
export type RunStatus = 'idle' | 'submitting' | 'queued' | 'loading-model' | 'running' | 'completed' | 'failed'

interface Form {
  model: string
  inputMode: 'text' | 'example'
  prompt: string
  exampleIndex: number
  numSamples: number
  seed: number
  sampling: Partial<SamplingParams>
}

interface PlaygroundState {
  form: Form
  setForm: (patch: Partial<Form>) => void
  setSampling: (patch: Partial<SamplingParams>) => void
  resetSampling: () => void

  status: RunStatus
  experimentId: string | null
  events: GenEvent[]
  levels: Record<number, LevelState>
  steps: number
  record: ExperimentRecord | null
  trace: Trace | null
  error: string | null
  run: (levelsSchedule: number[], conditional: boolean) => Promise<void>
}

let closeStream: (() => void) | null = null

export const usePlayground = create<PlaygroundState>((set, get) => ({
  form: { model: '', inputMode: 'text', prompt: '', exampleIndex: 0, numSamples: 2, seed: 0, sampling: {} },
  setForm: (patch) => set((s) => {
    const switching = patch.model !== undefined && patch.model !== s.form.model
    const idle = !['submitting', 'queued', 'loading-model', 'running'].includes(s.status)
    // results belong to the previous model's level schedule: drop them on switch
    const cleared = switching && idle
      ? { status: 'idle' as RunStatus, events: [], levels: {}, steps: 0, record: null, trace: null, error: null, experimentId: null }
      : {}
    return { form: { ...s.form, ...patch }, ...cleared }
  }),
  setSampling: (patch) => set((s) => ({ form: { ...s.form, sampling: { ...s.form.sampling, ...patch } } })),
  resetSampling: () => set((s) => ({ form: { ...s.form, sampling: {} } })),

  status: 'idle',
  experimentId: null,
  events: [],
  levels: {},
  steps: 0,
  record: null,
  trace: null,
  error: null,

  run: async (levelsSchedule, conditional) => {
    closeStream?.()
    const { form } = get()
    const body: GenerateRequest = {
      model: form.model,
      num_samples: form.numSamples,
      seed: form.seed,
      sampling: form.sampling,
      include_trace: true,
    }
    if (conditional) {
      if (form.inputMode === 'example') body.example_index = form.exampleIndex
      else body.prompt = form.prompt
    }
    const pending: Record<number, LevelState> = Object.fromEntries(
      levelsSchedule.map((_, i): [number, LevelState] => [i, { status: 'pending', seconds: null }]))
    set({ status: 'submitting', events: [], levels: pending, steps: 0, record: null, trace: null, error: null, experimentId: null })

    let id: string
    try {
      id = (await api.createExperiment(body)).id
    } catch (e) {
      set({ status: 'failed', error: (e as Error).message })
      return
    }
    set({ experimentId: id, status: 'queued' })

    closeStream = streamExperiment(id, {
      onEvent: (ev) => {
        set((s) => {
          const levels = { ...s.levels }
          let status = s.status
          if (ev.type === 'generation.started') status = 'running'
          if (ev.type === 'model.loading') status = 'loading-model'
          if (ev.type === 'model.loaded') status = 'running'
          if (ev.type === 'generation.level_started' && ev.level !== undefined)
            levels[ev.level] = { status: 'running', seconds: null }
          if (ev.type === 'generation.level_completed' && ev.level !== undefined)
            levels[ev.level] = { status: 'done', seconds: ev.seconds ?? null, source: ev.source }
          if (ev.type === 'generation.failed') return { events: [...s.events, ev], status: 'failed', error: ev.error ?? 'failed' }
          const steps = ev.type === 'generation.step_completed' && ev.step !== undefined ? ev.step : s.steps
          return { events: [...s.events, ev], levels, status, steps }
        })
        if (ev.type === 'generation.completed') void finish(id)
      },
      onClose: (code) => {
        if (get().experimentId !== id) return
        const st = get().status
        if (st === 'completed' || st === 'failed') return
        // completion is fetched asynchronously after generation.completed
        if (get().events.some((e) => e.type === 'generation.completed')) return
        set({ status: 'failed', error: code === 4404 ? 'experiment not found' : 'connection to the backend closed before the run finished' })
      },
    })

    async function finish(eid: string) {
      try {
        const [record, trace] = await Promise.all([api.experiment(eid), api.trace(eid).catch(() => null)])
        if (get().experimentId === eid) set({ record, trace, status: 'completed' })
      } catch (e) {
        set({ status: 'failed', error: (e as Error).message })
      }
    }
  },
}))
