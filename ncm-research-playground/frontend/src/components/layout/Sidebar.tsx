import {
  Activity,
  BarChart3,
  Boxes,
  FlaskConical,
  GraduationCap,
  GitCompareArrows,
  History,
  LayoutGrid,
  PlayCircle,
  Settings,
  type LucideIcon,
} from 'lucide-react'
import { NavLink } from 'react-router-dom'

import { StatusDot } from '@/components/common'
import { cn } from '@/lib/utils'
import { useModels } from '@/store/models'

type Item = { to: string; label: string; icon: LucideIcon }

const GROUPS: { label: string; items: Item[] }[] = [
  { label: 'Workspace', items: [
    { to: '/learn', label: 'Learn NCM', icon: GraduationCap },
    { to: '/', label: 'Overview', icon: LayoutGrid },
    { to: '/how-it-works', label: 'How NCM works', icon: PlayCircle },
    { to: '/playground', label: 'Playground', icon: FlaskConical },
  ] },
  { label: 'Analysis', items: [
    { to: '/concepts', label: 'Concept Explorer', icon: Boxes },
    { to: '/timeline', label: 'Generation Timeline', icon: Activity },
    { to: '/compare', label: 'Model Comparison', icon: GitCompareArrows },
    { to: '/benchmarks', label: 'Benchmarks', icon: BarChart3 },
  ] },
  { label: 'Records', items: [{ to: '/history', label: 'Experiment History', icon: History }] },
]

function Link({ item }: { item: Item }) {
  const Icon = item.icon
  return (
    <NavLink
      to={item.to}
      end={item.to === '/'}
      title={item.label}
      className={({ isActive }) => cn(
        'group flex items-center gap-2.5 rounded-md px-2.5 py-1.5 text-[0.84rem] transition-colors',
        isActive
          ? 'bg-sidebar-accent text-sidebar-accent-foreground'
          : 'text-sidebar-foreground/70 hover:bg-sidebar-accent/60 hover:text-sidebar-accent-foreground',
      )}
    >
      {({ isActive }) => (
        <>
          <Icon className={cn('size-4 shrink-0', isActive ? 'text-primary' : 'opacity-70')} strokeWidth={1.75} />
          <span className="hidden truncate md:inline">{item.label}</span>
        </>
      )}
    </NavLink>
  )
}

export function Sidebar() {
  const { status, device, error } = useModels()
  const tone = status === 'ready' ? 'ok' : status === 'error' ? 'error' : 'busy'
  return (
    <aside className="sticky top-0 flex h-svh w-14 shrink-0 flex-col border-r border-sidebar-border bg-sidebar md:w-60">
      <div className="flex h-14 items-center gap-2.5 border-b border-sidebar-border px-3.5 md:px-4">
        <svg viewBox="0 0 20 20" className="size-5 shrink-0 text-primary" aria-hidden>
          <rect x="1" y="2" width="18" height="3" rx="1" fill="currentColor" opacity="0.35" />
          <rect x="1" y="8.5" width="12" height="3" rx="1" fill="currentColor" opacity="0.65" />
          <rect x="1" y="15" width="6" height="3" rx="1" fill="currentColor" />
        </svg>
        <div className="hidden leading-tight md:block">
          <div className="text-sm font-semibold tracking-tight">NCM Playground</div>
          <div className="text-[0.7rem] text-muted-foreground">Next-Concept Modeling</div>
        </div>
      </div>
      <nav className="flex-1 space-y-5 overflow-y-auto px-2 py-4">
        {GROUPS.map((g) => (
          <div key={g.label}>
            <div className="mb-1.5 hidden px-2.5 text-[0.68rem] font-medium tracking-wider text-muted-foreground/70 uppercase md:block">
              {g.label}
            </div>
            <div className="space-y-0.5">{g.items.map((it) => <Link key={it.to} item={it} />)}</div>
          </div>
        ))}
      </nav>
      <div className="space-y-1 border-t border-sidebar-border px-2 py-3">
        <Link item={{ to: '/settings', label: 'Settings', icon: Settings }} />
        <div className="flex items-center gap-2 px-2.5 py-1 text-[0.72rem] text-muted-foreground" title={error ?? undefined}>
          <StatusDot tone={tone} />
          <span className="hidden truncate md:inline">
            {status === 'ready' ? `backend · ${device}` : status === 'error' ? 'backend offline' : 'connecting…'}
          </span>
        </div>
      </div>
    </aside>
  )
}
