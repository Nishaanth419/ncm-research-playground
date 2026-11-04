import { ArrowRight, BookOpen, ExternalLink, FileText, Lightbulb, PlayCircle, Video } from 'lucide-react'
import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'

import { PageHeader, Panel, StatTile } from '@/components/common'
import { Button } from '@/components/ui/button'
import { CAVEATS, FINDINGS, GLOSSARY, KEY_IDEAS, PAPER, VIDEOS } from '@/features/learn/content'
import { VideoCard } from '@/features/learn/VideoCard'
import { cn } from '@/lib/utils'

export function Learn() {
  return (
    <>
      <PageHeader
        title="Learn NCM"
        description="What Next-Concept Modeling is, what the research paper shows, what we found when we ran it, and videos to go further. No background needed."
      />
      <div className="mx-auto max-w-5xl space-y-10 p-4 sm:p-6 lg:p-8">
        <WhatIsIt />

        <Section icon={Lightbulb} title="The key ideas, in plain words">
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {KEY_IDEAS.map((k, i) => (
              <div key={k.title} className="rounded-xl border border-border bg-card p-4">
                <div className="mb-1.5 flex items-center gap-2">
                  <span className="flex size-5 items-center justify-center rounded-full bg-primary/15 font-mono text-[0.68rem] text-primary">{i + 1}</span>
                  <span className="text-sm font-medium">{k.title}</span>
                </div>
                <p className="text-sm leading-relaxed text-muted-foreground">{k.body}</p>
              </div>
            ))}
          </div>
        </Section>

        <Section icon={FileText} title="The research paper">
          <Panel>
            <div className="text-base font-medium">{PAPER.title}</div>
            <div className="mt-1 text-xs text-muted-foreground">{PAPER.authors} · {PAPER.year}</div>
            <p className="mt-3 text-sm leading-relaxed">
              The paper proposes a two-stage recipe for generating text in parallel. First, a tokenizer learns to describe any passage as a coarse-to-fine hierarchy of discrete codes.
              Second, a transformer learns to generate that hierarchy one level at a time, all positions of a level together, so the number of sequential steps is the number of levels rather than the number of words.
            </p>
            <p className="mt-2 text-sm leading-relaxed">
              It is tested on story generation, open-domain text, summarisation, translation and Sudoku.
              The authors report that NCM matches or beats strong <em>diffusion</em> language models while generating <strong className="text-ncm">9–31× faster</strong> (measured on an H100 GPU).
            </p>
            <div className="mt-3 flex flex-wrap gap-1.5">
              {PAPER.tasks.map((t) => <span key={t} className="rounded-md border border-border px-2 py-0.5 text-xs text-muted-foreground">{t}</span>)}
            </div>
            <div className="mt-4 flex flex-wrap gap-2">
              {PAPER.links.map((l, i) => (
                <Button key={l.href} size="sm" variant={i === 0 ? 'default' : 'outline'} nativeButton={false}
                  render={<a href={l.href} target="_blank" rel="noreferrer" />}>
                  {l.label}<ExternalLink />
                </Button>
              ))}
            </div>
          </Panel>
        </Section>

        <Section icon={BookOpen} title="What we found running it">
          <p className="mb-3 text-sm text-muted-foreground">
            We ran the official code and checkpoints on a laptop CPU (Apple M5 Pro) and compared NCM with TinyStories-33M, a traditional word-by-word model trained on the same stories.
          </p>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {FINDINGS.map((f) => <StatTile key={f.label} label={f.label} value={f.value} hint={f.hint} />)}
          </div>
          <div className="mt-4 rounded-xl border border-border bg-card/60 p-4">
            <div className="mb-2 text-xs font-medium tracking-wide text-muted-foreground uppercase">What this does and doesn&rsquo;t show</div>
            <ul className="list-disc space-y-1 pl-5 text-sm text-muted-foreground">
              {CAVEATS.map((c) => <li key={c}>{c}</li>)}
            </ul>
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            <Button size="sm" variant="outline" nativeButton={false} render={<Link to="/compare" />}>Run the comparison yourself <ArrowRight /></Button>
            <Button size="sm" variant="outline" nativeButton={false} render={<Link to="/benchmarks" />}>See the benchmark <ArrowRight /></Button>
          </div>
        </Section>

        <Section icon={Video} title="Videos">
          <p className="mb-3 text-sm text-muted-foreground">
            Explainers by other creators. They are useful introductions, but they are not reviewed by us; when they disagree with the paper, trust the paper.
            Nothing loads from YouTube except the thumbnails until you press play.
          </p>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {VIDEOS.map((v) => <VideoCard key={v.id} video={v} />)}
          </div>
        </Section>

        <Section icon={BookOpen} title="Glossary">
          <dl className="grid grid-cols-1 gap-x-6 gap-y-3 sm:grid-cols-2">
            {GLOSSARY.map((g) => (
              <div key={g.term} className="border-l-2 border-primary/40 pl-3">
                <dt className="text-sm font-medium">{g.term}</dt>
                <dd className="text-sm text-muted-foreground">{g.def}</dd>
              </div>
            ))}
          </dl>
        </Section>
      </div>
    </>
  )
}

function Section({ icon: Icon, title, children }: { icon: typeof BookOpen; title: string; children: ReactNode }) {
  return (
    <section className="min-w-0">
      <h2 className="mb-3 flex items-center gap-2 text-lg font-semibold tracking-tight">
        <Icon className="size-4.5 text-primary" strokeWidth={1.75} />{title}
      </h2>
      {children}
    </section>
  )
}

function WhatIsIt() {
  return (
    <section className="min-w-0">
      <h2 className="mb-3 text-lg font-semibold tracking-tight">What is NCM?</h2>
      <div className="space-y-2 text-[0.95rem] leading-relaxed">
        <p>
          Chatbots like ChatGPT write the way you type a message: <span className="text-ar">one word at a time</span>, left to right. To write 128 words they run their network up to 128 times, one after another, and each run has to wait for the previous one.
        </p>
        <p>
          <strong>Next-Concept Modeling (NCM)</strong>, from Stanford&rsquo;s Hazy Research lab, writes the way you might draft an essay: <span className="text-ncm">outline first, then detail</span>.
          It first decides the gist of the whole passage, then sharpens every part of it at once, a few rounds in a row. A 128-word story takes 11 rounds instead of 128.
        </p>
      </div>
      <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-2">
        <Sketch tone="ar" title="Word by word (traditional)" caption="Each step adds one word at the end.">
          {[1, 2, 3, 4, 5].map((n) => (
            <Row key={n}>{Array.from({ length: 8 }, (_, i) => <Cell key={i} on={i < n} tone="ar" />)}</Row>
          ))}
        </Sketch>
        <Sketch tone="ncm" title="Coarse to fine (NCM)" caption="Each step covers the whole passage, in more detail.">
          {[1, 2, 4, 8, 16].map((n) => (
            <Row key={n}>{Array.from({ length: n }, (_, i) => <Cell key={i} on tone="ncm" shade={(i * 7 + n) % 5} />)}</Row>
          ))}
        </Sketch>
      </div>
      <p className="mt-2 text-xs text-muted-foreground">Schematic. To see a real run play out level by level:</p>
      <Button size="sm" className="mt-2" nativeButton={false} render={<Link to="/how-it-works" />}><PlayCircle />Watch How NCM works</Button>
    </section>
  )
}

function Sketch({ tone, title, caption, children }: { tone: 'ar' | 'ncm'; title: string; caption: string; children: ReactNode }) {
  return (
    <div className="rounded-xl border border-border bg-card p-4">
      <div className={cn('mb-2 text-sm font-medium', tone === 'ar' ? 'text-ar' : 'text-ncm')}>{title}</div>
      <div className="space-y-1">{children}</div>
      <div className="mt-2 text-xs text-muted-foreground">{caption}</div>
    </div>
  )
}

function Row({ children }: { children: ReactNode }) {
  return <div className="flex h-3 gap-0.5">{children}</div>
}

function Cell({ on, tone, shade = 0 }: { on: boolean; tone: 'ar' | 'ncm'; shade?: number }) {
  if (!on) return <span className="flex-1 rounded-[2px] border border-dashed border-border" />
  const pct = tone === 'ncm' ? 45 + shade * 12 : 85
  return <span className="flex-1 rounded-[2px]" style={{ background: `color-mix(in oklch, var(--${tone}) ${pct}%, transparent)` }} />
}
