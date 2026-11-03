/** Autoregressive progress: one sequential forward pass per generated token. */
export function StepProgress({ steps, max, done }: { steps: number; max: number; done: boolean }) {
  const pct = Math.min(100, (steps / Math.max(1, max)) * 100)
  return (
    <div>
      <div className="relative h-12 overflow-hidden rounded-[3px] border border-border bg-muted/40">
        <div className="absolute inset-y-0 left-0 bg-ar/30 transition-[width] duration-150" style={{ width: `${pct}%` }} />
        {/* tick every 16 steps to make "one step per token" visible */}
        <div className="absolute inset-0 flex">
          {Array.from({ length: Math.ceil(max / 16) }, (_, i) => (
            <div key={i} className="flex-1 border-r border-background/60 last:border-0" />
          ))}
        </div>
      </div>
      <div className="mt-2 flex justify-between text-[0.7rem] text-muted-foreground">
        <span>token 1</span>
        <span className="font-mono tabular">{steps}{done && steps < max ? ' (EOS)' : ''} / {max} steps</span>
        <span>token {max}</span>
      </div>
    </div>
  )
}
