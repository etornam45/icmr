import { useCallback, useEffect, useState } from 'react'
import {
  fetchEvents,
  startSource,
  stopSource,
  uploadSource,
  type EventItem,
  type SourceType,
} from './api'
import { EventTimeline } from './components/EventTimeline'
import { InferenceGrid } from './components/InferenceGrid'
import { SourceControls } from './components/SourceControls'
import { connectLive, type LiveFrameMessage } from './live'
import './App.css'

function App() {
  const [detectionUrl, setDetectionUrl] = useState<string | null>(null)
  const [pcaUrl, setPcaUrl] = useState<string | null>(null)
  const [anomalyUrl, setAnomalyUrl] = useState<string | null>(null)
  const [isAnomaly, setIsAnomaly] = useState(false)
  const [running, setRunning] = useState(false)
  const [busy, setBusy] = useState(false)
  const [events, setEvents] = useState<EventItem[]>([])
  const [wsStatus, setWsStatus] = useState<'connecting' | 'open' | 'closed'>(
    'connecting',
  )

  const refreshEvents = useCallback(async () => {
    try {
      const list = await fetchEvents()
      setEvents(list)
    } catch {
      // server may be starting
    }
  }, [])

  useEffect(() => {
    void refreshEvents()
    const disconnect = connectLive({
      onStatus: setWsStatus,
      onMessage: (msg) => {
        if (msg.type === 'frame') {
          applyFrame(msg)
        } else if (msg.type === 'event') {
          setEvents((prev) => {
            const rest = prev.filter((e) => e.id !== msg.event.id)
            return [msg.event as EventItem, ...rest]
          })
        }
      },
    })
    return disconnect
  }, [refreshEvents])

  const applyFrame = (msg: LiveFrameMessage) => {
    const { frames } = msg
    setDetectionUrl(`data:image/jpeg;base64,${frames.detection}`)
    setPcaUrl(`data:image/jpeg;base64,${frames.pca}`)
    setAnomalyUrl(`data:image/jpeg;base64,${frames.anomaly}`)
    setIsAnomaly(msg.is_anomaly)
    setRunning(msg.running)
  }

  const clearFrames = () => {
    setDetectionUrl(null)
    setPcaUrl(null)
    setAnomalyUrl(null)
    setIsAnomaly(false)
  }

  const onStart = async (type: SourceType, uri: string) => {
    setBusy(true)
    try {
      await startSource(type, uri)
      setRunning(true)
    } finally {
      setBusy(false)
    }
  }

  const onUpload = async (file: File) => {
    setBusy(true)
    try {
      await uploadSource(file)
      setRunning(true)
    } finally {
      setBusy(false)
    }
  }

  const onStop = async () => {
    setBusy(true)
    try {
      await stopSource()
      setRunning(false)
      clearFrames()
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="app-shell">
      <header className="top-bar">
        <div className="brand">
          <span className="brand-name">ICMR</span>
          <span className="brand-sub">Intelligent CCTV Monitoring</span>
        </div>
        <div className={`ws-pill ${wsStatus}`}>
          {wsStatus === 'open' ? 'Live link' : wsStatus}
        </div>
      </header>

      <main className="monitor">
        <section className="stage">
          <InferenceGrid
            detectionUrl={detectionUrl}
            pcaUrl={pcaUrl}
            anomalyUrl={anomalyUrl}
            isAnomaly={isAnomaly}
            events={events}
            idle={!running && !detectionUrl}
          />
          <div className="stage-controls">
            <SourceControls
              busy={busy}
              running={running}
              onStart={onStart}
              onUpload={onUpload}
              onStop={onStop}
            />
          </div>
        </section>

        <aside className="rail">
          <EventTimeline events={events} />
        </aside>
      </main>
    </div>
  )
}

export default App
