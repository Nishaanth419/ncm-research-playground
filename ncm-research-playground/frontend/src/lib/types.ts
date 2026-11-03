// Mirrors backend/app/schemas/*.py. Keep in sync with the Pydantic models.

export type PromptKind = 'none' | 'article' | 'german'

export interface SamplingParams {
  temperature: number
  top_k: number
  top_p: number
  cfg_scale: number
}

export interface ModelInfo {
  id: string
  family: 'ncm' | 'autoregressive'
  task: string
  conditional: boolean
  prompt_kind: PromptKind
  description: string
  available: boolean
  missing_files: string[]
  loaded: boolean
  parameters: Record<string, number> | null
  levels: number[]
  output_tokens: number
  prefix_tokens: number
  n_examples: number
  defaults: SamplingParams
}

export interface GenerateRequest {
  model: string
  prompt?: string
  example_index?: number
  num_samples: number
  seed: number
  sampling: Partial<SamplingParams>
  include_trace: boolean
}

export interface Timing {
  load_s: number
  generate_s: number
  decode_s: number
  total_s: number
}

export interface CodebookStats {
  vocab_size: number
  n_codes: number
  unique: number
  utilization: number
  entropy_bits: number
  perplexity: number
  top: { code: number; count: number }[]
}

export interface LatentSummary {
  norm_mean: number
  delta_norm_mean: number
  cos_to_final: number
  n_samples: number
}

export interface LevelTrace {
  level: number
  n_codes: number
  source: 'unigram' | 'transformer' | 'given'
  logits_shape: number[] | null
  seconds: number | null
  stats: CodebookStats
  latent_in: LatentSummary | null
  codes: number[][]
  decoded: string[]
}

export interface Trace {
  task: string
  device: string
  batch_size: number
  levels_schedule: number[]
  transformer_passes: number
  total_seconds: number
  latent_samples: number
  max_detail_samples: number
  levels: LevelTrace[]
  final_texts: string[]
}

export type ExperimentStatus = 'queued' | 'running' | 'completed' | 'failed'

export interface GenEvent {
  type:
    | 'generation.started'
    | 'model.loading'
    | 'model.loaded'
    | 'generation.level_started'
    | 'generation.level_completed'
    | 'generation.completed'
    | 'generation.failed'
  experiment_id: string
  seq: number
  t: number
  replay?: boolean
  level?: number
  n_codes?: number
  seconds?: number | null
  source?: string
  outputs?: string[]
  timing?: Timing
  error?: string
  [k: string]: unknown
}

export interface ExperimentMetrics {
  model: string
  task: string
  device: string
  num_samples: number
  timing: Timing
  transformer_passes: number
  levels: number[]
  per_level: { level: number; n_codes: number; source: string; seconds: number | null; unique: number; entropy_bits: number }[]
  output_words: number[]
  output_chars: number[]
  sampling: SamplingParams
  prompt_tokens: number | null
  prompt_truncated: boolean
  reference: string | null
}

export interface ExperimentRecord {
  id: string
  created_at: string
  finished_at: string | null
  model: string
  status: ExperimentStatus
  error: string | null
  request: GenerateRequest
  outputs: string[] | null
  timing: Timing | null
  metrics?: ExperimentMetrics | null
  events?: GenEvent[] | null
}
