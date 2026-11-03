export function fmtSeconds(s: number | null | undefined): string {
  if (s === null || s === undefined || Number.isNaN(s)) return '—'
  if (s < 1) return `${(s * 1000).toFixed(s < 0.01 ? 2 : 1)} ms`
  return `${s.toFixed(2)} s`
}

export function fmtInt(n: number | null | undefined): string {
  return n === null || n === undefined ? '—' : n.toLocaleString('en-US')
}

export function fmtParams(n: number): string {
  return n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : `${(n / 1e3).toFixed(0)}K`
}

export function fmtRelative(iso: string): string {
  const d = (Date.now() - new Date(iso).getTime()) / 1000
  if (d < 60) return `${Math.max(0, Math.round(d))}s ago`
  if (d < 3600) return `${Math.round(d / 60)}m ago`
  if (d < 86400) return `${Math.round(d / 3600)}h ago`
  return new Date(iso).toLocaleDateString()
}
