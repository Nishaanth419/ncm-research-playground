import { describe, expect, it } from 'vitest'

import { chartRows, medianRatio } from './bench'
import type { BenchmarkGroup, Stat } from './types'

const st = (median: number, p95: number): Stat => ({ n: 3, median, p90: p95, p95, mean: median, std: 0, min: median, max: p95 })
const g = (model: string, bs: number, med: number, p95: number): BenchmarkGroup => ({
  model, batch_size: bs, n: 3, generate_s: st(med, p95), total_s: st(med, p95), first_output_s: st(med, p95),
  per_sample_s: st(med / bs, p95 / bs), words_per_s: st(100 / med, 100 / p95), tokens_per_s: st(1, 1), passes: 11,
  mean_output_tokens: 128, memory_peak_bytes: null,
})

describe('chartRows', () => {
  it('groups by batch size with median bars and p95 whiskers', () => {
    const rows = chartRows([g('a', 4, 2, 3), g('b', 1, 1, 1.5), g('a', 1, 0.5, 0.6)], 'total_s')
    expect(rows.map((r) => r.batch)).toEqual(['batch 1', 'batch 4'])
    expect(rows[0].a).toBe(0.5)
    expect(rows[0]['a__err']).toEqual([0, expect.closeTo(0.1)])
    expect(rows[1].b).toBeUndefined()
  })
})

describe('medianRatio', () => {
  it('divides medians for the same batch size', () => {
    const s = [g('a', 1, 1, 1), g('b', 1, 2, 2)]
    expect(medianRatio(s, 'b', 'a', 1, 'total_s')).toBe(2)
    expect(medianRatio(s, 'b', 'a', 4, 'total_s')).toBeNull()
  })
})
