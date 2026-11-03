import type { ExperimentMetrics, ExperimentRecord, GenEvent, GenerateRequest, ModelInfo, QualityResult, Trace } from './types'

export class ApiError extends Error {
  readonly status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(path, { headers: { 'content-type': 'application/json' }, ...init })
  } catch {
    throw new ApiError(0, 'Backend unreachable. Is uvicorn running on :8000?')
  }
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail)
  }
  return res.json() as Promise<T>
}

export const api = {
  health: () => request<{ status: string; device: string }>('/api/health'),
  models: () => request<ModelInfo[]>('/api/models'),
  createExperiment: (body: GenerateRequest) =>
    request<{ id: string; status: string; stream_url: string }>('/api/experiments', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  experiments: (limit = 50) => request<ExperimentRecord[]>(`/api/experiments?limit=${limit}`),
  experiment: (id: string) => request<ExperimentRecord>(`/api/experiments/${id}`),
  trace: (id: string) => request<Trace>(`/api/experiments/${id}/trace`),
  metrics: (id: string) => request<ExperimentMetrics>(`/api/experiments/${id}/metrics`),
  quality: (id: string) => request<QualityResult>(`/api/experiments/${id}/quality`, { method: 'POST' }),
}

/** Subscribe to the real event stream of one experiment. Returns a closer. */
export function streamExperiment(
  id: string,
  handlers: { onEvent: (e: GenEvent) => void; onClose?: (code: number) => void },
): () => void {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
  const ws = new WebSocket(`${proto}://${window.location.host}/api/stream/${id}`)
  ws.onmessage = (m) => handlers.onEvent(JSON.parse(m.data) as GenEvent)
  ws.onclose = (e) => handlers.onClose?.(e.code)
  return () => ws.close()
}

/** Create an experiment, relay its real events, resolve with the finished record. */
export async function runExperiment(body: GenerateRequest, onEvent: (e: GenEvent) => void): Promise<ExperimentRecord> {
  const { id } = await api.createExperiment(body)
  await new Promise<void>((resolve, reject) => {
    const close = streamExperiment(id, {
      onEvent: (e) => {
        onEvent(e)
        if (e.type === 'generation.completed') { close(); resolve() }
        if (e.type === 'generation.failed') { close(); reject(new Error(e.error ?? 'generation failed')) }
      },
      onClose: (code) => { if (code === 4404) reject(new Error('experiment not found')) },
    })
  })
  return api.experiment(id)
}
