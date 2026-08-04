import { wsUrl } from './api'

export type LiveFrameMessage = {
  type: 'frame'
  jpeg_b64: string
  anomaly: string | null
  score: number | null
  top_k?: { class: string; probability: number }[]
  overlay_mode: string
  detections: unknown[]
  source: { type: string; uri: string } | null
  running: boolean
  is_anomaly: boolean
}

export type LiveEventMessage = {
  type: 'event'
  event: {
    id: number
    timestamp: number
    iso_time: string
    source: string
    anomaly_class: string
    score: number
    caption: string | null
  }
}

export type LiveErrorMessage = {
  type: 'error'
  message: string
}

export type LiveMessage = LiveFrameMessage | LiveEventMessage | LiveErrorMessage

type Handlers = {
  onMessage: (msg: LiveMessage) => void
  onStatus?: (status: 'connecting' | 'open' | 'closed') => void
}

export function connectLive(handlers: Handlers): () => void {
  let ws: WebSocket | null = null
  let closed = false
  let pingTimer: number | undefined
  let retryTimer: number | undefined

  const connect = () => {
    if (closed) return
    handlers.onStatus?.('connecting')
    ws = new WebSocket(wsUrl())

    ws.onopen = () => {
      handlers.onStatus?.('open')
      pingTimer = window.setInterval(() => {
        if (ws?.readyState === WebSocket.OPEN) ws.send('ping')
      }, 20000)
    }

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data) as LiveMessage
        handlers.onMessage(msg)
      } catch {
        // ignore malformed
      }
    }

    ws.onclose = () => {
      handlers.onStatus?.('closed')
      if (pingTimer) window.clearInterval(pingTimer)
      if (!closed) {
        retryTimer = window.setTimeout(connect, 1500)
      }
    }

    ws.onerror = () => {
      ws?.close()
    }
  }

  connect()

  return () => {
    closed = true
    if (pingTimer) window.clearInterval(pingTimer)
    if (retryTimer) window.clearTimeout(retryTimer)
    ws?.close()
  }
}
