import { useCallback, useEffect, useRef, useState } from 'react'
import {
  fetchEvents,
  setOverlay,
  startSource,
  stopSource,
  uploadSource,
  type EventItem,
  type OverlayMode,
  type SourceType,
} from './api'
import { EventTimeline } from './components/EventTimeline'
import { LiveView } from './components/LiveView'
import { OverlayToggle } from './components/OverlayToggle'
import { SourceControls } from './components/SourceControls'
import { connectLive, type LiveFrameMessage } from './live'
import './App.css'

function App() {
  const [frameUrl, setFrameUrl] = useState<string | null>(null)
  const [anomaly, setAnomaly] = useState<string | null>(null)
  const [score, setScore] = useState<number | null>(null)
  const [isAnomaly, setIsAnomaly] = useState(false)
  const [running, setRunning] = useState(false)
  const [busy, setBusy] = useState(false)
  const [overlay, setOverlayMode] = useState<OverlayMode>('none')
  const [events, setEvents] = useState<EventItem[]>([])
  const [wsStatus, setWsStatus] = useState<'connecting' | 'open' | 'closed'>(
    'connecting',
  )
  const prevUrl = useRef<string | null>(null)

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
    const url = `data:image/jpeg;base64,${msg.jpeg_b64}`
    if (prevUrl.current) URL.revokeObjectURL(prevUrl.current)
    // data URLs don't need revoke, but keep ref for consistency if we switch
    prevUrl.current = null
    setFrameUrl(url)
    setAnomaly(msg.anomaly)
    setScore(msg.score)
    setIsAnomaly(msg.is_anomaly)
    setRunning(msg.running)
    if (
      msg.overlay_mode === 'none' ||
      msg.overlay_mode === 'detection' ||
      msg.overlay_mode === 'pca'
    ) {
      setOverlayMode(msg.overlay_mode)
    }
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
      setFrameUrl(null)
      setAnomaly(null)
      setScore(null)
      setIsAnomaly(false)
    } finally {
      setBusy(false)
    }
  }

  const onOverlay = async (mode: OverlayMode) => {
    setOverlayMode(mode)
    try {
      await setOverlay(mode)
    } catch {
      // ignore
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
          <LiveView
            frameUrl={frameUrl}
            anomaly={anomaly}
            score={score}
            isAnomaly={isAnomaly}
          />
          <div className="stage-controls">
            <SourceControls
              busy={busy}
              running={running}
              onStart={onStart}
              onUpload={onUpload}
              onStop={onStop}
            />
            <OverlayToggle mode={overlay} onChange={onOverlay} />
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
