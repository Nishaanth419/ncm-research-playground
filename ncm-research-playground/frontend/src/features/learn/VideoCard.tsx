import { ExternalLink, Play } from 'lucide-react'
import { useState } from 'react'

import type { Video } from './content'

/**
 * Thumbnail first; the YouTube player (privacy-enhanced domain) is only loaded
 * when the viewer clicks play, so opening the page contacts YouTube for images only.
 */
export function VideoCard({ video }: { video: Video }) {
  const [playing, setPlaying] = useState(false)
  const watch = `https://www.youtube.com/watch?v=${video.id}`
  return (
    <div className="min-w-0 overflow-hidden rounded-xl border border-border bg-card">
      <div className="relative aspect-video bg-muted">
        {playing ? (
          <iframe
            className="absolute inset-0 size-full"
            src={`https://www.youtube-nocookie.com/embed/${video.id}?autoplay=1`}
            title={video.title}
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowFullScreen
          />
        ) : (
          <button type="button" onClick={() => setPlaying(true)} aria-label={`Play: ${video.title}`}
            className="group absolute inset-0 size-full">
            <img src={`https://i.ytimg.com/vi/${video.id}/hqdefault.jpg`} alt="" loading="lazy"
              className="size-full object-cover opacity-85 transition-opacity group-hover:opacity-100" />
            <span className="absolute inset-0 flex items-center justify-center">
              <span className="flex size-12 items-center justify-center rounded-full bg-black/70 text-white transition-transform group-hover:scale-110">
                <Play className="ml-0.5 size-5 fill-current" />
              </span>
            </span>
          </button>
        )}
      </div>
      <div className="p-3">
        <div className="line-clamp-2 text-sm font-medium">{video.title}</div>
        <div className="mt-1 flex items-center justify-between gap-2 text-xs text-muted-foreground">
          <span className="truncate">{video.channel} · {video.note}</span>
          <a href={watch} target="_blank" rel="noreferrer" className="flex shrink-0 items-center gap-1 hover:text-foreground">
            YouTube <ExternalLink className="size-3" />
          </a>
        </div>
      </div>
    </div>
  )
}
