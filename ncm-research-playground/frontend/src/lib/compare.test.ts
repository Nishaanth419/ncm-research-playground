import { describe, expect, it } from 'vitest'

import { caveats, summarize } from './compare'
import type { ExperimentMetrics, ModelInfo } from './types'

const sp = { temperature: 1, top_k: 0, top_p: 1, cfg_scale: 0 }
const model = (over: Partial<ModelInfo>): ModelInfo => ({
  id: 'm', family: 'ncm', task: 'tinystories', conditional: false, prompt_kind: 'none', description: '',
  available: true, missing_files: [], loaded: true, parameters: null, levels: [], output_tokens: 128,
  prefix_tokens: 1, n_examples: 0, defaults: sp, tokenizer: 't', training_data: 'TinyStories', notes: [], ...over,
})
const metrics = (over: Partial<ExperimentMetrics>): ExperimentMetrics => ({
  model: 'm', task: 'tinystories', device: 'cpu', num_samples: 2,
  timing: { load_s: 0, generate_s: 1, decode_s: 0.1, total_s: 1.1, first_output_s: 1.1 },
  transformer_passes: 11, levels: [], per_level: [], output_words: [10, 20], output_chars: [50, 100], sampling: sp,
  prompt_tokens: null, prompt_truncated: false, reference: null, output_tokens: [128, 64], tokenizer: 't',
  throughput_tokens_per_s: 192, throughput_words_per_s: 27.3, output_entropy_own: [4, 3], memory: null, quality: null,
  ...over,
})

describe('summarize', () => {
  it('averages per sample and keeps missing quality as null', () => {
    const s = summarize(metrics({}))
    expect(s.meanTokens).toBe(96)
    expect(s.meanWords).toBe(15)
    expect(s.genPpl).toBeNull()
    expect(s.entropyOwn).toBe(3.5)
  })

  it('averages judge entropy only over scored samples', () => {
    const q = { judge: 'gpt2-large', protocol: '', corpus_ppl: 12, n_scored: 1,
      per_sample: [{ ppl: 12, judge_tokens: 30, entropy_judge_tokens: 4 }, null] }
    const s = summarize(metrics({ quality: q }))
    expect(s.genPpl).toBe(12)
    expect(s.entropyJudge).toBe(4)
  })
})

describe('caveats', () => {
  const ncm = model({ id: 'ncm', family: 'ncm', parameters: { generator: 88e6, tokenizer_vqvae: 36e6 }, tokenizer: 'ncm-bpe' })
  const ar = model({ id: 'ar', family: 'autoregressive', parameters: { model: 68e6 }, tokenizer: 'gpt2-bpe',
    training_data: 'TinyStories (independent run)' })

  it('flags parameter, tokenizer and training-data differences', () => {
    const c = caveats(ncm, ar, metrics({}), metrics({}))
    expect(c.some((x) => x.includes('Parameter counts differ'))).toBe(true)
    expect(c.some((x) => x.includes('Different tokenizers'))).toBe(true)
    expect(c.some((x) => x.includes('Training data'))).toBe(true)
    expect(c.some((x) => x.includes('Single run'))).toBe(true)
    expect(c.some((x) => x.includes('without a warm-up'))).toBe(true)
  })

  it('states the warm-up when both runs had one', () => {
    const c = caveats(ncm, ar, metrics({ warmup: true }), metrics({ warmup: true }))
    expect(c.some((x) => x.includes('after one untimed warm-up pass each'))).toBe(true)
  })

  it('flags mismatched decoding settings and sample counts', () => {
    const c = caveats(ncm, ar, metrics({}), metrics({ sampling: { ...sp, temperature: 0.7 }, num_samples: 4 }))
    expect(c.some((x) => x.includes('Decoding settings differ'))).toBe(true)
    expect(c.some((x) => x.includes('different number of samples'))).toBe(true)
  })

  it('does not claim a tokenizer difference when there is none', () => {
    const c = caveats(ncm, { ...ar, tokenizer: 'ncm-bpe' }, metrics({}), metrics({}))
    expect(c.some((x) => x.includes('Different tokenizers'))).toBe(false)
  })
})
