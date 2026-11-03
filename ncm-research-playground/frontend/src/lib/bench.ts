import type { BenchmarkGroup, Stat } from './types'

type StatKey = 'total_s' | 'generate_s' | 'per_sample_s' | 'words_per_s' | 'first_output_s'

/**
 * One row per batch size; per model `<model>` = median and `<model>__err` =
 * [0, p95 - median] (Recharts ErrorBar takes deltas): bar = median, whisker to p95.
 */
export function chartRows(summary: BenchmarkGroup[], key: StatKey): Record<string, number | string | [number, number]>[] {
  const sizes = [...new Set(summary.map((g) => g.batch_size))].sort((a, b) => a - b)
  return sizes.map((bs) => {
    const row: Record<string, number | string | [number, number]> = { batch: `batch ${bs}` }
    for (const g of summary.filter((x) => x.batch_size === bs)) {
      const s: Stat = g[key]
      if (s.median === null) continue
      row[g.model] = s.median
      row[`${g.model}__err`] = [0, Math.max(0, (s.p95 ?? s.median) - s.median)]
    }
    return row
  })
}

/** Ratio of medians a/b for the same batch size, or null when not measured. */
export function medianRatio(summary: BenchmarkGroup[], a: string, b: string, bs: number, key: StatKey): number | null {
  const ga = summary.find((g) => g.model === a && g.batch_size === bs)?.[key].median
  const gb = summary.find((g) => g.model === b && g.batch_size === bs)?.[key].median
  return ga && gb ? ga / gb : null
}
