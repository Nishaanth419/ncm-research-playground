import type { GenEvent } from './types'

export interface Segment {
  kind: 'load' | 'level' | 'decode'
  level?: number
  n_codes?: number
  source?: string
  start: number // seconds since generation.started (event `t`)
  end: number
  incomplete?: boolean
}

/**
 * Lay out the recorded execution on one time axis, using only the event
 * timestamps the backend emitted:
 *  - load:  model.loading → model.loaded
 *  - level: level_started → level_completed. Level 0 has no started event
 *           (unigram draw, or predicted before the first hook); a transformer
 *           level 0 is placed using its measured `seconds`, a unigram draw is a
 *           zero-width marker.
 *  - decode: last level_completed → generation.completed
 */
export function timelineSegments(events: GenEvent[]): Segment[] {
  const segs: Segment[] = []
  const open = new Map<number, GenEvent>()
  let loadStart: number | null = null
  let lastLevelEnd: number | null = null
  for (const e of events) {
    if (e.type === 'model.loading') loadStart = e.t
    else if (e.type === 'model.loaded' && loadStart !== null) {
      segs.push({ kind: 'load', start: loadStart, end: e.t })
      loadStart = null
    } else if (e.type === 'generation.level_started' && e.level !== undefined) open.set(e.level, e)
    else if (e.type === 'generation.level_completed' && e.level !== undefined) {
      const st = open.get(e.level)
      open.delete(e.level)
      const start = st ? st.t : e.seconds != null ? Math.max(0, e.t - e.seconds) : e.t
      segs.push({ kind: 'level', level: e.level, n_codes: e.n_codes, source: e.source, start, end: e.t })
      lastLevelEnd = e.t
    } else if (e.type === 'generation.completed' && lastLevelEnd !== null) {
      segs.push({ kind: 'decode', start: lastLevelEnd, end: e.t })
    } else if (e.type === 'generation.failed') {
      for (const st of open.values())
        segs.push({ kind: 'level', level: st.level, n_codes: st.n_codes, start: st.t, end: e.t, incomplete: true })
      open.clear()
    }
  }
  return segs
}

/** Occurrences of each code id across the stored detail samples of one level. */
export function codeCounts(codes: number[][]): Map<number, number> {
  const m = new Map<number, number>()
  for (const row of codes) for (const c of row) m.set(c, (m.get(c) ?? 0) + 1)
  return m
}

/**
 * Categorical colour for a code id (golden-angle hue spacing). Same id → same
 * colour; colours carry no notion of similarity between codes.
 */
export function codeColor(code: number): string {
  const hue = (code * 137.508) % 360
  return `oklch(0.68 0.11 ${hue.toFixed(1)})`
}
