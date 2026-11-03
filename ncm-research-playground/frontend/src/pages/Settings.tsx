import { KV, PageHeader, Panel } from '@/components/common'
import { useModels } from '@/store/models'

export function SettingsPage() {
  const { status, device, error, models } = useModels()
  return (
    <>
      <PageHeader title="Settings" description="Backend connection and runtime facts. Server-side settings come from environment variables." />
      <div className="grid gap-6 p-4 sm:p-6 lg:grid-cols-2 lg:p-8">
        <Panel title="Backend">
          <KV k="API" v="/api (proxied by Vite to NCM_BACKEND_URL, default http://127.0.0.1:8000)" />
          <KV k="Status" v={status === 'ready' ? 'connected' : status === 'error' ? `error: ${error}` : status} />
          <KV k="Device" v={device ?? '—'} />
          <KV k="Checkpoints on disk" v={`${models.filter((m) => m.available).length} / ${models.length}`} />
        </Panel>
        <Panel title="Server environment variables">
          <KV k="NCM_DEVICE" v="cpu (MPS fails on non-divisible area pooling; see docs/ASSESSMENT.md)" />
          <KV k="NCM_DB_PATH" v="results/playground.sqlite" />
          <KV k="NCM_RESEARCH_ROOT" v="research-ncm/ (official repo, pinned SHA)" />
          <KV k="NCM_CORS_ORIGINS" v="http://localhost:5173" />
        </Panel>
      </div>
    </>
  )
}
