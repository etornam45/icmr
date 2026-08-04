import { useState } from 'react'
import type { EventItem } from '../api'

type Props = {
  events: EventItem[]
}

export function EventTimeline({ events }: Props) {
  const [openId, setOpenId] = useState<number | null>(null)

  if (events.length === 0) {
    return (
      <div className="timeline empty">
        <h2>Events</h2>
        <p>No anomalies recorded yet.</p>
      </div>
    )
  }

  return (
    <div className="timeline">
      <h2>Events</h2>
      <ul>
        {events.map((ev) => {
          const open = openId === ev.id
          return (
            <li key={ev.id}>
              <button
                type="button"
                className="event-row"
                onClick={() => setOpenId(open ? null : ev.id)}
              >
                <span className="event-time">{ev.iso_time}</span>
                <span className="event-class">{ev.anomaly_class}</span>
                <span className="event-score">{ev.score.toFixed(2)}</span>
              </button>
              {open && (
                <p className="event-caption">
                  {ev.caption ?? 'Caption pending…'}
                </p>
              )}
            </li>
          )
        })}
      </ul>
    </div>
  )
}
