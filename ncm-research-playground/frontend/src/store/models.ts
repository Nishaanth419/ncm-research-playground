import { create } from 'zustand'

import { api } from '@/lib/api'
import type { ModelInfo } from '@/lib/types'

interface ModelsState {
  models: ModelInfo[]
  status: 'idle' | 'loading' | 'ready' | 'error'
  error: string | null
  device: string | null
  refresh: () => Promise<void>
}

export const useModels = create<ModelsState>((set) => ({
  models: [],
  status: 'idle',
  error: null,
  device: null,
  refresh: async () => {
    set({ status: 'loading', error: null })
    try {
      const [models, health] = await Promise.all([api.models(), api.health()])
      set({ models, device: health.device, status: 'ready' })
    } catch (e) {
      set({ status: 'error', error: (e as Error).message })
    }
  },
}))
