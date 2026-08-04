import type { EventItem } from '../api'

type FramePaneProps = {
  title: string
  frameUrl: string | null
  alert?: boolean
  emptyHint?: string
}

function FramePane({ title, frameUrl, alert = false, emptyHint }: FramePaneProps) {
  return (
    <div className={`inference-pane ${alert ? 'alert' : ''}`}>
      <div className="pane-title">{title}</div>
      <div className="pane-body">
        {frameUrl ? (
          <img src={frameUrl} alt={`${title} preview`} className="live-frame" />
        ) : (
          <div className="live-placeholder pane-placeholder">
            <p>{emptyHint ?? 'Waiting for frames…'}</p>
          </div>
        )}
      </div>
    </div>
  )
}

type CaptionsPaneProps = {
  events: EventItem[]
}

function CaptionsPane({ events }: CaptionsPaneProps) {
  return (
    <div className="inference-pane captions-pane">
      <div className="pane-title">Captions</div>
      <div className="pane-body captions-body">
        {events.length === 0 ? (
          <div className="captions-empty">
            <p>Captions appear when an anomaly is detected.</p>
          </div>
        ) : (
          <div className="captions-list">
            {events.map((ev) => {
              const pending = ev.caption == null
              return (
                <article key={ev.id} className="captions-item">
                  <div className="captions-meta">
                    <span className="captions-class">{ev.anomaly_class}</span>
                    <span className="captions-score">{ev.score.toFixed(2)}</span>
                    <span className="captions-time">{ev.iso_time}</span>
                  </div>
                  <p className="captions-text">
                    {pending ? 'Caption pending…' : ev.caption}
                  </p>
                </article>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}

type Props = {
  detectionUrl: string | null
  pcaUrl: string | null
  anomalyUrl: string | null
  isAnomaly: boolean
  events: EventItem[]
  idle: boolean
}

export function InferenceGrid({
  detectionUrl,
  pcaUrl,
  anomalyUrl,
  isAnomaly,
  events,
  idle,
}: Props) {
  const hint = idle
    ? 'Start a file or stream source to begin monitoring'
    : 'Waiting for frames…'

  return (
    <div className="inference-grid">
      <FramePane title="Detection" frameUrl={detectionUrl} emptyHint={hint} />
      <FramePane title="PCA" frameUrl={pcaUrl} emptyHint={hint} />
      <FramePane
        title="Anomaly"
        frameUrl={anomalyUrl}
        alert={isAnomaly}
        emptyHint={hint}
      />
      <CaptionsPane events={events} />
    </div>
  )
}
