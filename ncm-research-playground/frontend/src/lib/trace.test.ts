import { describe, expect, it } from 'vitest'

import { codeColor, codeCounts, timelineSegments } from './trace'
import type { GenEvent } from './types'

const ev = (seq: number, t: number, type: GenEvent['type'], extra: Partial<GenEvent> = {}): GenEvent =>
  ({ experiment_id: 'e', seq, t, type, ...extra }) as GenEvent

describe('timelineSegments', () => {
  it('builds load, unigram level 0, transformer levels and decode from real events', () => {
    const events = [
      ev(0, 0.0, 'generation.started'),
      ev(1, 0.01, 'model.loading'),
      ev(2, 2.0, 'model.loaded', { seconds: 1.99 }),
      ev(3, 2.1, 'generation.level_completed', { level: 0, n_codes: 1, seconds: null, source: 'unigram' }),
      ev(4, 2.1, 'generation.level_started', { level: 1, n_codes: 8 }),
      ev(5, 2.3, 'generation.level_completed', { level: 1, n_codes: 8, seconds: 0.2, source: 'transformer' }),
      ev(6, 2.3, 'generation.level_started', { level: 2, n_codes: 16 }),
      ev(7, 2.6, 'generation.level_completed', { level: 2, n_codes: 16, seconds: 0.3, source: 'transformer' }),
      ev(8, 2.65, 'generation.completed'),
    ]
    const segs = timelineSegments(events)
    expect(segs.map((s) => [s.kind, s.level ?? null])).toEqual([
      ['load', null], ['level', 0], ['level', 1], ['level', 2], ['decode', null],
    ])
    expect(segs[0].start).toBeCloseTo(0.01)
    expect(segs[0].end).toBeCloseTo(2.0)
    expect(segs[1].end - segs[1].start).toBe(0) // unigram draw: no transformer step
    expect(segs[2].start).toBeCloseTo(2.1)
    expect(segs[2].end).toBeCloseTo(2.3)
    expect(segs[4].start).toBeCloseTo(2.6)
    expect(segs[4].end).toBeCloseTo(2.65)
  })

  it('places a transformer-predicted level 0 (no started event) using its measured seconds', () => {
    const events = [
      ev(0, 0, 'generation.started'),
      ev(1, 0.5, 'generation.level_completed', { level: 0, n_codes: 4, seconds: 0.4, source: 'transformer' }),
      ev(2, 0.5, 'generation.level_started', { level: 1, n_codes: 6 }),
      ev(3, 0.6, 'generation.level_completed', { level: 1, n_codes: 6, seconds: 0.1, source: 'transformer' }),
      ev(4, 0.61, 'generation.completed'),
    ]
    const segs = timelineSegments(events)
    expect(segs[0]).toMatchObject({ kind: 'level', level: 0, source: 'transformer' })
    expect(segs[0].start).toBeCloseTo(0.1)
    expect(segs[0].end).toBeCloseTo(0.5)
  })

  it('stops at the last completed level when generation failed', () => {
    const events = [
      ev(0, 0, 'generation.started'),
      ev(1, 0.1, 'generation.level_started', { level: 1, n_codes: 8 }),
      ev(2, 0.2, 'generation.failed', { error: 'x' }),
    ]
    const segs = timelineSegments(events)
    expect(segs).toHaveLength(1)
    expect(segs[0]).toMatchObject({ kind: 'level', level: 1, incomplete: true })
  })
})

describe('code helpers', () => {
  it('counts codes across detail samples', () => {
    const c = codeCounts([[1, 2, 2], [2, 3, 1]])
    expect(c.get(2)).toBe(3)
    expect(c.get(1)).toBe(2)
    expect(c.get(9)).toBeUndefined()
  })

  it('gives the same code the same colour and different codes different hues', () => {
    expect(codeColor(17)).toBe(codeColor(17))
    expect(codeColor(17)).not.toBe(codeColor(18))
  })
})
