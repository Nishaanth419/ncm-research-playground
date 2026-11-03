import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

import type { Trace } from '@/lib/types'

const axis = { fill: 'var(--muted-foreground)', fontSize: 10, fontFamily: 'var(--font-mono)' }

function Mini({ title, data, dataKey, selected, domain, digits }: {
  title: string
  data: Record<string, number | null>[]
  dataKey: string
  selected: number
  domain?: [number, number]
  digits: number
}) {
  return (
    <div className="min-w-0">
      <div className="mb-1 text-xs text-muted-foreground">{title}</div>
      <div className="h-32">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ left: -18, right: 8, top: 6, bottom: 0 }}>
            <CartesianGrid stroke="var(--border)" vertical={false} />
            <XAxis dataKey="level" tick={axis} tickLine={false} axisLine={false} interval={0} tickFormatter={(l) => `${l}`} />
            <YAxis tick={axis} tickLine={false} axisLine={false} domain={domain ?? ['auto', 'auto']} tickFormatter={(v: number) => v.toFixed(digits)} width={48} />
            <ReferenceLine x={selected} stroke="var(--ncm)" strokeOpacity={0.5} />
            <Tooltip
              contentStyle={{ background: 'var(--popover)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }}
              labelFormatter={(l) => `level ${l}`}
              formatter={(v) => [typeof v === 'number' ? v.toFixed(digits + 1) : '—', title]}
            />
            <Line dataKey={dataKey} stroke="var(--ncm)" strokeWidth={1.75} dot={{ r: 2 }} isAnimationActive={false} connectNulls={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

export function LevelSeries({ trace, selected }: { trace: Trace; selected: number }) {
  const data = trace.levels.map((lv) => ({
    level: lv.level,
    utilization: lv.stats.utilization * 100,
    entropy: lv.stats.entropy_bits,
    cos: lv.latent_in?.cos_to_final ?? null,
  }))
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
      <Mini title="codebook utilization %" data={data} dataKey="utilization" selected={selected} digits={1} />
      <Mini title="code entropy (bits)" data={data} dataKey="entropy" selected={selected} digits={1} />
      <Mini title="cos(input latent, final)" data={data} dataKey="cos" selected={selected} domain={[0, 1]} digits={2} />
    </div>
  )
}
