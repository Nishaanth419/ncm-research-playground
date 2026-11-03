import type { LucideIcon } from 'lucide-react'
import type { ReactNode } from 'react'

import { cn } from '@/lib/utils'

export function PageHeader({ title, description, actions }: { title: string; description?: ReactNode; actions?: ReactNode }) {
  return (
    <header className="flex flex-wrap items-end justify-between gap-4 border-b border-border px-4 pt-7 pb-5 sm:px-6 lg:px-8">
      <div className="min-w-0">
        <h1 className="text-[1.35rem] font-semibold tracking-tight">{title}</h1>
        {description && <p className="mt-1 max-w-3xl text-sm text-muted-foreground">{description}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </header>
  )
}

export function Panel({ title, meta, children, className, bodyClassName }: {
  title?: ReactNode
  meta?: ReactNode
  children: ReactNode
  className?: string
  bodyClassName?: string
}) {
  return (
    <section className={cn('min-w-0 rounded-xl border border-border bg-card', className)}>
      {(title || meta) && (
        <div className="flex items-center justify-between gap-3 border-b border-border px-4 py-2.5">
          <h2 className="text-[0.8rem] font-medium tracking-wide text-muted-foreground uppercase">{title}</h2>
          {meta && <div className="text-xs text-muted-foreground">{meta}</div>}
        </div>
      )}
      <div className={cn('p-4', bodyClassName)}>{children}</div>
    </section>
  )
}

export function StatTile({ label, value, hint }: { label: string; value: ReactNode; hint?: ReactNode }) {
  return (
    <div className="min-w-0 rounded-xl border border-border bg-card px-4 py-3.5">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="mt-1.5 font-mono text-xl tracking-tight tabular sm:text-2xl">{value}</div>
      {hint && <div className="mt-1 text-xs break-words text-muted-foreground">{hint}</div>}
    </div>
  )
}

export function EmptyState({ icon: Icon, title, children, className }: {
  icon: LucideIcon
  title: string
  children?: ReactNode
  className?: string
}) {
  return (
    <div className={cn('flex flex-col items-center justify-center rounded-xl border border-dashed border-border px-6 py-12 text-center', className)}>
      <Icon className="size-5 text-muted-foreground" strokeWidth={1.5} />
      <div className="mt-3 text-sm font-medium">{title}</div>
      {children && <div className="mt-1 max-w-md text-sm text-muted-foreground">{children}</div>}
    </div>
  )
}

export function StatusDot({ tone }: { tone: 'ok' | 'warn' | 'error' | 'idle' | 'busy' }) {
  const cls = {
    ok: 'bg-ok',
    warn: 'bg-primary',
    error: 'bg-destructive',
    idle: 'bg-muted-foreground/50',
    busy: 'bg-primary animate-pulse',
  }[tone]
  return <span className={cn('inline-block size-1.5 shrink-0 rounded-full', cls)} />
}

export function KV({ k, v }: { k: ReactNode; v: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-1 text-sm">
      <span className="shrink-0 text-muted-foreground">{k}</span>
      <span className="min-w-0 text-right font-mono text-[0.8rem] break-words tabular">{v}</span>
    </div>
  )
}
