import type { Trace } from './types'

/** What happens for one level, in order: predict its codes, add them to the latent, decode. */
export type Phase = 'predict' | 'add' | 'decode'
export const PHASES: Phase[] = ['predict', 'add', 'decode']

export interface Step {
  level: number
  phase: Phase
}

/** Playback order for the coarse-to-fine replay: three phases per level. */
export function playbackSteps(nLevels: number): Step[] {
  const steps: Step[] = []
  for (let level = 0; level < nLevels; level++) for (const phase of PHASES) steps.push({ level, phase })
  return steps
}

/**
 * Levels whose codes are on the canvas at this step: every earlier level, plus
 * the current one once it has been predicted (it is shown from 'predict' on).
 */
export function visibleLevels(step: Step): number {
  return step.level + 1
}

/** Index of the level whose decode is shown: the current level once decoded, else the previous one (-1 = none yet). */
export function decodedLevel(step: Step): number {
  return step.phase === 'decode' ? step.level : step.level - 1
}

export interface RaceState {
  arTokens: number // tokens the autoregressive model has emitted
  ncmLevels: number // NCM levels on the canvas
  arDone: boolean
  ncmDone: boolean
}

/**
 * Both models after `passes` forward passes. The autoregressive model emits one
 * token per pass, up to the canvas length. NCM completes one level per pass; a
 * level 0 drawn from the unigram (TinyStories) costs no pass and is there at 0.
 */
export function raceState(passes: number, trace: Pick<Trace, 'levels' | 'levels_schedule'>): RaceState {
  const length = trace.levels_schedule[trace.levels_schedule.length - 1] ?? 0
  const free = trace.levels[0]?.source === 'unigram' ? 1 : 0
  const nLevels = trace.levels.length
  const arTokens = Math.min(length, Math.max(0, passes))
  const ncmLevels = Math.min(nLevels, free + Math.max(0, passes))
  return { arTokens, ncmLevels, arDone: arTokens === length, ncmDone: ncmLevels === nLevels }
}

/** Forward passes NCM needs for the whole canvas (levels minus a free unigram level 0). */
export function ncmPasses(trace: Pick<Trace, 'levels'>): number {
  return trace.levels.length - (trace.levels[0]?.source === 'unigram' ? 1 : 0)
}
