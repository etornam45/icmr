const API_BASE = import.meta.env.VITE_API_BASE ?? ''

export type SourceType = 'file' | 'rtsp' | 'stream'

export type EventItem = {
  id: number
  timestamp: number
  iso_time: string
  source: string
  anomaly_class: string
  score: number
  caption: string | null
  thumbnail_path: string | null
}

export async function startSource(type: SourceType, uri: string) {
  const res = await fetch(`${API_BASE}/source`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ type, uri }),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function uploadSource(file: File) {
  const body = new FormData()
  body.append('file', file)
  const res = await fetch(`${API_BASE}/source/upload`, {
    method: 'POST',
    body,
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function stopSource() {
  const res = await fetch(`${API_BASE}/source/stop`, { method: 'POST' })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function fetchEvents(limit = 50): Promise<EventItem[]> {
  const res = await fetch(`${API_BASE}/events?limit=${limit}`)
  if (!res.ok) throw new Error(await res.text())
  const data = await res.json()
  return data.events as EventItem[]
}

export function wsUrl(): string {
  if (import.meta.env.VITE_WS_BASE) {
    return `${import.meta.env.VITE_WS_BASE}/ws/live`
  }
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
  return `${proto}://${window.location.host}/ws/live`
}

export { API_BASE }
