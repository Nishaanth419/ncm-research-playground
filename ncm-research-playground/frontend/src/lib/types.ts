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
  tokenizer: string
  training_data: string
  notes: string[]
}

export interface GenerateRequest {
  model: string
  prompt?: string
  example_index?: number
  num_samples: number
  seed: number
  sampling: Partial<SamplingParams>
  include_trace: boolean
  tag?: string
  warmup?: boolean
}

export interface Timing {
  load_s: number
  generate_s: number
  decode_s: number
  total_s: number
  first_output_s?: number | null
}

export interface MemoryStats {
  device: string
  peak_bytes: number | null
  note: string
}

export interface QualityResult {
  judge: string
  protocol: string
  corpus_ppl: number | null
  n_scored: number
  per_sample: ({ ppl: number; judge_tokens: number; entropy_judge_tokens: number } | null)[]
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
    | 'generation.step_completed'
    | 'generation.completed'
    | 'generation.failed'
  experiment_id: string
  seq: number
  t: number
  replay?: boolean
  level?: number
  step?: number
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
  output_tokens: number[]
  tokenizer: string
  throughput_tokens_per_s: number | null
  throughput_words_per_s: number | null
  output_entropy_own: number[]
  memory: MemoryStats | null
  quality: QualityResult | null
  warmup?: boolean
  tag?: string | null
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
