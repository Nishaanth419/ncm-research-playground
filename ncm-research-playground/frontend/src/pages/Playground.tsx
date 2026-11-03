import { Dices, FlaskConical, Loader2, Play } from 'lucide-react'
import { useEffect, useMemo } from 'react'
import { Link } from 'react-router-dom'

import { EmptyState, KV, PageHeader, Panel, StatusDot } from '@/components/common'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Textarea } from '@/components/ui/textarea'
import { EventLog } from '@/features/playground/EventLog'
import { LevelStrip } from '@/features/playground/LevelStrip'
import { ParamControl } from '@/features/playground/ParamControl'
import { fmtSeconds } from '@/lib/format'
import { useModels } from '@/store/models'
import { usePlayground, type RunStatus } from '@/store/playground'

const STATUS_TEXT: Record<RunStatus, string> = {
  idle: 'Idle',
  submitting: 'Submitting…',
  queued: 'Queued: waiting for the inference worker',
  'loading-model': 'Loading checkpoint into memory…',
  running: 'Generating level by level…',
  completed: 'Completed',
  failed: 'Failed',
}

const PROMPT_LABEL = { article: 'News article (first 512 GPT-2 tokens are used)', german: 'German source sentence (first 96 Qwen3 tokens are used)', none: '' }

export function Playground() {
  const models = useModels((s) => s.models)
  const { form, setForm, setSampling, resetSampling, status, events, levels, record, error, run, experimentId } = usePlayground()
  const model = useMemo(() => models.find((m) => m.id === form.model), [models, form.model])

  useEffect(() => {
    if (!form.model && models.length) setForm({ model: (models.find((m) => m.available) ?? models[0]).id })
  }, [models, form.model, setForm])

  const busy = ['submitting', 'queued', 'loading-model', 'running'].includes(status)
  const needsText = model?.conditional && form.inputMode === 'text' && !form.prompt.trim()
  const canRun = !!model?.available && !busy && !needsText
  const d = model?.defaults
  const greedy = (form.sampling.top_k ?? d?.top_k) === 1

  return (
    <>
      <PageHeader
        title="Playground"
        description="Run real NCM inference with the official checkpoints. Every status update below is an execution event from the backend."
      />
      <div className="grid grid-cols-1 gap-6 p-4 sm:p-6 lg:grid-cols-[380px_minmax(0,1fr)] lg:p-8">
        {/* ---------------- configuration ---------------- */}
        <div className="space-y-4">
          <Panel title="Model">
            <Select value={form.model} onValueChange={(v) => { if (v) { setForm({ model: v as string }); resetSampling() } }}>
              <SelectTrigger className="w-full font-mono text-[0.82rem]"><SelectValue placeholder="Choose a model" /></SelectTrigger>
              <SelectContent>
                {models.map((m) => (
                  <SelectItem key={m.id} value={m.id} disabled={!m.available} className="font-mono text-[0.82rem]">{m.id}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            {model && (
              <div className="mt-3 space-y-2">
                <p className="text-xs leading-relaxed text-muted-foreground">{model.description}</p>
                <div className="flex flex-wrap gap-1.5">
                  <Badge variant="outline" className="font-normal">{model.levels.length} levels</Badge>
                  <Badge variant="outline" className="font-normal">{model.levels.reduce((a, b) => a + b, 0)} codes</Badge>
                  <Badge variant="outline" className="font-normal">{model.output_tokens} output tokens</Badge>
                  {!model.available && <Badge variant="destructive">checkpoint missing</Badge>}
                </div>
              </div>
            )}
          </Panel>

          {model?.conditional && (
            <Panel title="Input">
              <Tabs value={form.inputMode} onValueChange={(v) => setForm({ inputMode: v as 'text' | 'example' })}>
                <TabsList className="w-full">
                  <TabsTrigger value="text">Free text</TabsTrigger>
                  <TabsTrigger value="example">Official test example</TabsTrigger>
                </TabsList>
              </Tabs>
              {form.inputMode === 'text' ? (
                <div className="mt-3 space-y-1.5">
                  <Label className="text-xs font-normal text-muted-foreground">{PROMPT_LABEL[model.prompt_kind]}</Label>
                  <Textarea
                    value={form.prompt}
                    onChange={(e) => setForm({ prompt: e.target.value })}
                    rows={model.prompt_kind === 'article' ? 9 : 3}
                    placeholder={model.prompt_kind === 'german' ? 'Das Wetter ist heute schön.' : 'Paste a news article…'}
                    className="font-mono text-[0.78rem]"
                  />
                </div>
              ) : (
                <div className="mt-3 flex items-center gap-3">
                  <Label className="text-xs font-normal text-muted-foreground">index</Label>
                  <Input
                    type="number" min={0} max={model.n_examples - 1} value={form.exampleIndex}
                    onChange={(e) => setForm({ exampleIndex: Math.max(0, Math.min(model.n_examples - 1, Number(e.target.value) || 0)) })}
                    className="h-8 w-28 font-mono text-xs"
                  />
                  <span className="text-xs text-muted-foreground">of {model.n_examples.toLocaleString()} (reference returned with output)</span>
                </div>
              )}
            </Panel>
          )}

          {model && d && (
            <Panel title="Sampling" meta="defaults from the shipped config">
              <div className="space-y-4">
                <ParamControl label="temperature" value={form.sampling.temperature} defaultValue={d.temperature}
                  min={0} max={2} step={0.05} onChange={(v) => setSampling({ temperature: v })}
                  hint={greedy ? 'no effect with top-k 1' : undefined} />
                <ParamControl label="top-k" hint="0 = off" value={form.sampling.top_k} defaultValue={d.top_k}
                  min={0} max={256} step={1} onChange={(v) => setSampling({ top_k: v })} />
                <ParamControl label="top-p" value={form.sampling.top_p} defaultValue={d.top_p}
                  min={0} max={1} step={0.01} onChange={(v) => setSampling({ top_p: v })} />
                {model.conditional && (
                  <ParamControl label="cfg scale" hint="classifier-free guidance" value={form.sampling.cfg_scale} defaultValue={d.cfg_scale}
                    min={0} max={3} step={0.1} onChange={(v) => setSampling({ cfg_scale: v })} />
                )}
                <div className="grid grid-cols-2 gap-3 pt-1">
                  <div className="space-y-1.5">
                    <Label className="text-[0.8rem] font-normal">seed</Label>
                    <div className="flex gap-1">
                      <Input type="number" min={0} value={form.seed} onChange={(e) => setForm({ seed: Math.max(0, Number(e.target.value) || 0) })}
                        className="h-8 font-mono text-xs tabular" />
                      <Button variant="outline" size="icon" title="random seed" onClick={() => setForm({ seed: Math.floor(Math.random() * 2 ** 31) })}>
                        <Dices />
                      </Button>
                    </div>
                  </div>
                  <div className="space-y-1.5">
                    <Label className="text-[0.8rem] font-normal">samples</Label>
                    <Input type="number" min={1} max={16} value={form.numSamples}
                      onChange={(e) => setForm({ numSamples: Math.max(1, Math.min(16, Number(e.target.value) || 1)) })}
                      className="h-8 font-mono text-xs tabular" />
                  </div>
                </div>
              </div>
            </Panel>
          )}

          <Button size="lg" className="h-10 w-full" disabled={!canRun} onClick={() => model && run(model.levels, model.conditional)}>
            {busy ? <Loader2 className="animate-spin" /> : <Play />}
            {busy ? 'Running' : 'Generate'}
          </Button>
          {needsText && <p className="text-xs text-muted-foreground">Enter input text, or switch to an official test example.</p>}
        </div>

        {/* ---------------- execution + output ---------------- */}
        <div className="min-w-0 space-y-4">
          <Panel
            title="Execution"
            meta={experimentId && <span className="font-mono">exp {experimentId.slice(0, 8)}</span>}
          >
            <div className="mb-4 flex items-center gap-2 text-sm">
              <StatusDot tone={status === 'completed' ? 'ok' : status === 'failed' ? 'error' : busy ? 'busy' : 'idle'} />
              <span>{STATUS_TEXT[status]}</span>
              {error && <span className="truncate text-destructive">· {error}</span>}
            </div>
            {model ? <LevelStrip schedule={model.levels} levels={levels} /> : <div className="h-12" />}
            <p className="mt-3 text-xs text-muted-foreground">
              Levels run sequentially, left to right. Each cell is one transformer step that predicts all of that level's codes at once; width is proportional to the level's code count.
              {model?.conditional ? ' Level 0 is predicted by the transformer and its time includes encoding the input.' : ' Level 0 is drawn from the checkpoint’s unigram, as in the official eval.'}
            </p>
          </Panel>

          <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_280px]">
            <Panel
              title="Outputs"
              meta={record && experimentId ? (
                <span className="flex items-center gap-3">
                  <Link to={`/concepts?id=${experimentId}`} className="text-primary hover:underline">concepts →</Link>
                  <Link to={`/timeline?id=${experimentId}`} className="text-primary hover:underline">timeline →</Link>
                </span>
              ) : undefined}
            >
              {record?.outputs ? (
                <div className="space-y-3">
                  {record.outputs.map((o, i) => (
                    <div key={i} className="rounded-lg border border-border bg-background/40 p-3">
                      <div className="mb-1.5 font-mono text-[0.68rem] text-muted-foreground">sample {i}</div>
                      <p className="text-[0.88rem] leading-relaxed whitespace-pre-wrap">{o.trim() ? o.replace(/^\n+/, '') : <span className="text-muted-foreground">(empty after EOS cut)</span>}</p>
                    </div>
                  ))}
                  {record.metrics?.prompt_truncated && (
                    <p className="text-xs text-primary">Input was truncated to the model’s prefix length ({model?.prefix_tokens} tokens).</p>
                  )}
                </div>
              ) : (
                <EmptyState icon={FlaskConical} title={busy ? 'Waiting for the final level…' : 'No output yet'} className="py-10">
                  {busy ? 'Text is decoded from the final latent once every level has been generated.' : 'Configure a model and press Generate.'}
                </EmptyState>
              )}
            </Panel>
            <div className="space-y-4">
              <Panel title="Timing">
                <KV k="checkpoint load" v={fmtSeconds(record?.timing?.load_s)} />
                <KV k="generate" v={fmtSeconds(record?.timing?.generate_s)} />
                <KV k="decode" v={fmtSeconds(record?.timing?.decode_s)} />
                <KV k="total (excl. load)" v={fmtSeconds(record?.timing?.total_s)} />
                <KV k="transformer passes" v={record?.metrics?.transformer_passes ?? '—'} />
                <KV k="device" v={record?.metrics?.device ?? '—'} />
              </Panel>
              {record?.metrics?.prompt_tokens != null && (
                <Panel title="Input">
                  <KV k="input tokens" v={record.metrics.prompt_tokens} />
                  <KV k="used" v={Math.min(record.metrics.prompt_tokens, model?.prefix_tokens ?? 0)} />
                </Panel>
              )}
            </div>
          </div>

          {record?.metrics?.reference && (
            <Panel title="Reference" meta={`official test example #${record.request.example_index}`}>
              <p className="text-[0.88rem] leading-relaxed text-muted-foreground">{record.metrics.reference}</p>
            </Panel>
          )}

          <Panel title="Event stream" meta="WS /api/stream/{id}">
            <EventLog events={events} />
          </Panel>
        </div>
      </div>
    </>
  )
}
