import { fmtParams } from './format'
import type { ExperimentMetrics, ModelInfo } from './types'

const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null)

export interface Summary {
  totalS: number
  generateS: number
  firstOutputS: number | null
  passes: number
  meanTokens: number | null
  meanWords: number | null
  tokensPerS: number | null
  wordsPerS: number | null
  memoryBytes: number | null
  memoryNote: string
  genPpl: number | null
  entropyJudge: number | null
  entropyOwn: number | null
}

export function summarize(m: ExperimentMetrics): Summary {
  const scored = (m.quality?.per_sample ?? []).filter((p): p is NonNullable<typeof p> => p !== null)
  return {
    totalS: m.timing.total_s,
    generateS: m.timing.generate_s,
    firstOutputS: m.timing.first_output_s ?? null,
    passes: m.transformer_passes,
    meanTokens: mean(m.output_tokens),
    meanWords: mean(m.output_words),
    tokensPerS: m.throughput_tokens_per_s,
    wordsPerS: m.throughput_words_per_s,
    memoryBytes: m.memory?.peak_bytes ?? null,
    memoryNote: m.memory?.note ?? '',
    genPpl: m.quality?.corpus_ppl ?? null,
    entropyJudge: mean(scored.map((p) => p.entropy_judge_tokens)),
    entropyOwn: mean(m.output_entropy_own),
  }
}

const total = (p: Record<string, number> | null) => (p ? Object.values(p).reduce((a, b) => a + b, 0) : null)

/** Every difference that makes the two numbers not like-for-like, stated plainly. */
export function caveats(a: ModelInfo, b: ModelInfo, ma: ExperimentMetrics, mb: ExperimentMetrics): string[] {
  const out: string[] = []
  const pa = total(a.parameters), pb = total(b.parameters)
  if (pa && pb && Math.abs(pa - pb) / Math.max(pa, pb) > 0.05)
    out.push(`Parameter counts differ: ${a.id} ${fmtParams(pa)} (${Object.entries(a.parameters!).map(([k, v]) => `${k} ${fmtParams(v)}`).join(' + ')}) vs ${b.id} ${fmtParams(pb)}.`)
  if (a.tokenizer !== b.tokenizer)
    out.push(`Different tokenizers (${a.tokenizer} vs ${b.tokenizer}): token counts, tokens/s and own-tokenizer entropy are not directly comparable; use words, words/s and judge-token entropy across models.`)
  if (a.training_data !== b.training_data)
    out.push(`Training data: ${a.id}: ${a.training_data}; ${b.id}: ${b.training_data}. Same corpus does not mean the same amount of training.`)
  const sa = ma.sampling, sb = mb.sampling
  if (sa.temperature !== sb.temperature || sa.top_k !== sb.top_k || sa.top_p !== sb.top_p)
    out.push(`Decoding settings differ: ${a.id} T=${sa.temperature} k=${sa.top_k} p=${sa.top_p}; ${b.id} T=${sb.temperature} k=${sb.top_k} p=${sb.top_p}.`)
  if (ma.num_samples !== mb.num_samples) out.push('The runs used a different number of samples (batch size), which changes latency.')
  out.push(`Output length policy: ${a.family === 'ncm' ? a.id : b.id} always fills a fixed ${a.family === 'ncm' ? a.output_tokens : b.output_tokens}-token canvas (cut at EOS); ${a.family === 'autoregressive' ? a.id : b.id} generates up to ${a.family === 'autoregressive' ? a.output_tokens : b.output_tokens} tokens and stops at EOS.`)
  out.push('“First output” means different things: the first generated token for the autoregressive model; the complete text for NCM, since text exists only after the final level is decoded.')
  const warm = ma.warmup && mb.warmup ? 'after one untimed warm-up pass each' : 'without a warm-up pass (the first call after loading is slower)'
  out.push(`Single run on ${ma.device}: one measurement each, ${warm}, no repeated trials. Use Benchmarks for medians and percentiles. Published NCM speedups were measured on an H100 against diffusion baselines and are not comparable to this.`)
  return out
}
