import { describe, expect, it } from 'vitest'

import { decodedLevel, ncmPasses, playbackSteps, raceState, visibleLevels } from './howitworks'
import type { LevelTrace } from './types'

const lv = (source: LevelTrace['source']) => ({ source }) as LevelTrace
// TinyStories: 12 levels, level 0 from the unigram, 128-token canvas
const tiny = {
  levels: [lv('unigram'), ...Array.from({ length: 11 }, () => lv('transformer'))],
  levels_schedule: [1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128],
}
// conditional (XSum-like): every level is a transformer pass
const cond = { levels: Array.from({ length: 3 }, () => lv('transformer')), levels_schedule: [1, 4, 16] }

describe('playbackSteps', () => {
  it('visits predict, add, decode for every level in order', () => {
    const steps = playbackSteps(2)
    expect(steps).toEqual([
      { level: 0, phase: 'predict' }, { level: 0, phase: 'add' }, { level: 0, phase: 'decode' },
      { level: 1, phase: 'predict' }, { level: 1, phase: 'add' }, { level: 1, phase: 'decode' },
    ])
  })

  it('shows the current level from predict on, its text only once decoded', () => {
    expect(visibleLevels({ level: 3, phase: 'predict' })).toBe(4)
    expect(decodedLevel({ level: 3, phase: 'add' })).toBe(2)
    expect(decodedLevel({ level: 3, phase: 'decode' })).toBe(3)
    expect(decodedLevel({ level: 0, phase: 'predict' })).toBe(-1)
  })
})

describe('raceState', () => {
  it('gives NCM its unigram level 0 for free and finishes after 11 passes', () => {
    expect(ncmPasses(tiny)).toBe(11)
    expect(raceState(0, tiny)).toEqual({ arTokens: 0, ncmLevels: 1, arDone: false, ncmDone: false })
    expect(raceState(11, tiny)).toEqual({ arTokens: 11, ncmLevels: 12, arDone: false, ncmDone: true })
    expect(raceState(128, tiny)).toEqual({ arTokens: 128, ncmLevels: 12, arDone: true, ncmDone: true })
    expect(raceState(500, tiny).arTokens).toBe(128)
  })

  it('counts a pass for a transformer level 0', () => {
    expect(ncmPasses(cond)).toBe(3)
    expect(raceState(0, cond).ncmLevels).toBe(0)
    expect(raceState(3, cond)).toMatchObject({ ncmLevels: 3, ncmDone: true, arTokens: 3, arDone: false })
  })
})
