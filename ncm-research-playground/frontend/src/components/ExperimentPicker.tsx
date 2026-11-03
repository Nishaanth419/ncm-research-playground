import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { fmtRelative } from '@/lib/format'
import type { ExperimentRecord } from '@/lib/types'

export function describeInput(r: ExperimentRecord): string {
  if (r.request.example_index != null) return `test #${r.request.example_index}`
  if (r.request.prompt) return `“${r.request.prompt.slice(0, 28)}${r.request.prompt.length > 28 ? '…' : ''}”`
  return `seed ${r.request.seed} · ${r.request.num_samples}×`
}

export function ExperimentPicker({ candidates, selected, onSelect }: {
  candidates: ExperimentRecord[] | null
  selected: string | null
  onSelect: (id: string) => void
}) {
  const label = (r: ExperimentRecord) => `${r.model} · ${describeInput(r)} · ${fmtRelative(r.created_at)}`
  const current = candidates?.find((c) => c.id === selected)
  return (
    <Select value={selected ?? ''} onValueChange={(v) => v && onSelect(v as string)}>
      <SelectTrigger className="w-[min(26rem,calc(100vw-6rem))] font-mono text-[0.78rem]">
        <SelectValue placeholder={candidates?.length === 0 ? 'no traced experiments' : 'choose experiment'}>
          {current ? label(current) : selected ? `exp ${selected.slice(0, 8)}` : undefined}
        </SelectValue>
      </SelectTrigger>
      <SelectContent>
        {(candidates ?? []).map((r) => (
          <SelectItem key={r.id} value={r.id} className="font-mono text-[0.78rem]">{label(r)}</SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}
