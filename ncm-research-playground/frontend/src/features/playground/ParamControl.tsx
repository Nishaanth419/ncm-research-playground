import { RotateCcw } from 'lucide-react'

import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import { cn } from '@/lib/utils'

/** Slider + exact numeric input. `value` undefined means "use checkpoint default". */
export function ParamControl({ label, hint, value, defaultValue, min, max, step, onChange, disabled }: {
  label: string
  hint?: string
  value: number | undefined
  defaultValue: number
  min: number
  max: number
  step: number
  onChange: (v: number | undefined) => void
  disabled?: boolean
}) {
  const v = value ?? defaultValue
  const overridden = value !== undefined && value !== defaultValue
  return (
    <div className={cn('space-y-2', disabled && 'opacity-50')}>
      <div className="flex items-center justify-between gap-2">
        <Label className="text-[0.8rem] font-normal">
          {label}
          {hint && <span className="ml-1.5 text-muted-foreground">{hint}</span>}
        </Label>
        <div className="flex items-center gap-1">
          {overridden && (
            <button
              type="button"
              title={`reset to checkpoint default (${defaultValue})`}
              onClick={() => onChange(undefined)}
              className="rounded p-1 text-muted-foreground hover:text-foreground"
            >
              <RotateCcw className="size-3" />
            </button>
          )}
          <Input
            type="number"
            value={v}
            min={min}
            max={max}
            step={step}
            disabled={disabled}
            onChange={(e) => {
              const n = Number(e.target.value)
              if (!Number.isNaN(n)) onChange(Math.min(max, Math.max(min, n)))
            }}
            className={cn('h-7 w-20 text-right font-mono text-xs tabular', overridden && 'border-primary/40')}
          />
        </div>
      </div>
      <Slider
        value={[v]}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onValueChange={(x) => onChange(Array.isArray(x) ? x[0] : (x as number))}
      />
    </div>
  )
}
