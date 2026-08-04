type Props = {
  frameUrl: string | null
  anomaly: string | null
  score: number | null
  isAnomaly: boolean
}

export function LiveView({ frameUrl, anomaly, score, isAnomaly }: Props) {
  return (
    <div className={`live-view ${isAnomaly ? 'alert' : ''}`}>
      {frameUrl ? (
        <img src={frameUrl} alt="Live CCTV preview" className="live-frame" />
      ) : (
        <div className="live-placeholder">
          <span className="brand-mark">ICMR</span>
          <p>Start a file or stream source to begin monitoring</p>
        </div>
      )}
      {anomaly && (
        <div className="live-badge" aria-live="polite">
          <span className="badge-label">{anomaly}</span>
          {score != null && <span className="badge-score">{score.toFixed(2)}</span>}
        </div>
      )}
    </div>
  )
}
