import { useEffect, useState } from 'react'

/**
 * A step counter that advances on a timer: index runs 0..length-1 and stops at
 * the end. Playing again from the end restarts from 0.
 */
export function usePlayer(length: number, stepMs: number) {
  const [index, setIndex] = useState(0)
  const [playing, setPlaying] = useState(false)
  const last = Math.max(0, length - 1)
  const at = Math.min(index, last)

  useEffect(() => {
    if (!playing) return
    const t = setInterval(() => {
      setIndex((i) => {
        if (i >= last) { setPlaying(false); return last }
        return i + 1
      })
    }, stepMs)
    return () => clearInterval(t)
  }, [playing, stepMs, last])

  return {
    index: at,
    playing,
    atEnd: at === last,
    toggle: () => {
      if (!playing && at === last) setIndex(0)
      setPlaying((p) => !p)
    },
    seek: (i: number) => { setPlaying(false); setIndex(Math.max(0, Math.min(last, i))) },
    reset: () => { setPlaying(false); setIndex(0) },
  }
}
