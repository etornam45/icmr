import { Maximize2, Minimize2 } from "lucide-react";
import type { EventItem } from "../api";
import { useEffect, useState } from "react";

type PaneId = "detection" | "pca" | "anomaly" | "captions";

type PaneChromeProps = {
  id: PaneId;
  title: string;
  expanded: PaneId | null;
  onToggle: (id: PaneId) => void;
};

function PaneTitle({ id, title, expanded, onToggle }: PaneChromeProps) {
  const isExpanded = expanded === id;
  return (
    <div className="pane-title">
      <span>{title}</span>
      <button
        type="button"
        className="pane-expand"
        onClick={() => onToggle(id)}
        aria-label={isExpanded ? "Collapse ${title}" : "Expand ${title}"}
        title={isExpanded ? "Collapse" : "Expand"}
      >
        {isExpanded ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
      </button>
    </div>
  );
}

function paneClass(id: PaneId, expanded: PaneId | null, extra = "") {
  return [
    "inference-pane",
    extra,
    expanded === id ? "is-expanded" : "",
    expanded !== null && expanded !== id ? "is-hidden" : "",
  ]
    .filter(Boolean)
    .join(" ");
}

type FramePaneProps = {
  id: PaneId;
  title: string;
  frameUrl: string | null;
  alert?: boolean;
  emptyHint?: string;
  expanded: PaneId | null;
  onToggle: (id: PaneId) => void;
};

function FramePane({
  id,
  title,
  frameUrl,
  alert = false,
  emptyHint,
  expanded,
  onToggle,
}: FramePaneProps) {
  return (
    <div className={paneClass(id, expanded, alert ? "alert" : "")}>
      <PaneTitle
        id={id}
        title={title}
        expanded={expanded}
        onToggle={onToggle}
      />
      <div className="pane-body">
        {frameUrl ? (
          <img src={frameUrl} alt={`${title} preview`} className="live-frame" />
        ) : (
          <div className="live-placeholder pane-placeholder">
            <p>{emptyHint ?? "Waiting for frames…"}</p>
          </div>
        )}
      </div>
    </div>
  );
}

type CaptionsPaneProps = {
  events: EventItem[];
  expanded: PaneId | null;
  onToggle: (id: PaneId) => void;
};

function CaptionsPane({ events, expanded, onToggle }: CaptionsPaneProps) {
  return (
    <div className={paneClass("captions", expanded, "captions-pane")}>
      <PaneTitle
        id="captions"
        title="captions"
        expanded={expanded}
        onToggle={onToggle}
      />
      <div className="pane-body captions-body">
        {events.length === 0 ? (
          <div className="captions-empty">
            <p>Captions appear when an anomaly is detected.</p>
          </div>
        ) : (
          <div className="captions-list">
            {events.map((ev) => {
              const pending = ev.caption == null;
              return (
                <article key={ev.id} className="captions-item">
                  <div className="captions-meta">
                    <span className="captions-class">{ev.anomaly_class}</span>
                    <span className="captions-score">
                      {ev.score.toFixed(2)}
                    </span>
                    <span className="captions-time">{ev.iso_time}</span>
                  </div>
                  <p className="captions-text">
                    {pending ? "Caption pending…" : ev.caption}
                  </p>
                </article>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

type Props = {
  detectionUrl: string | null;
  pcaUrl: string | null;
  anomalyUrl: string | null;
  isAnomaly: boolean;
  events: EventItem[];
  idle: boolean;
};

export function InferenceGrid({
  detectionUrl,
  pcaUrl,
  anomalyUrl,
  isAnomaly,
  events,
  idle,
}: Props) {
  const [expanded, setExpanded] = useState<PaneId | null>(null);

  const toggle = (id: PaneId) => {
    setExpanded((prev) => (prev === id ? null : id));
  };

  useEffect(() => {
    if (expanded === null) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setExpanded(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [expanded]);

  const hint = idle
    ? "Start a file or stream source to begin monitoring"
    : "Waiting for frames…";

  return (
    <div className={`inference-grid ${expanded ? "has-expanded" : ""}`}>
      <FramePane
        id="detection"
        title="Detection"
        frameUrl={detectionUrl}
        emptyHint={hint}
        expanded={expanded}
        onToggle={toggle}
      />
      <FramePane
        id="pca"
        expanded={expanded}
        onToggle={toggle}
        title="PCA"
        frameUrl={pcaUrl}
        emptyHint={hint}
      />
      <FramePane
        id="anomaly"
        expanded={expanded}
        onToggle={toggle}
        title="Anomaly"
        frameUrl={anomalyUrl}
        alert={isAnomaly}
        emptyHint={hint}
      />
      <CaptionsPane expanded={expanded} onToggle={toggle} events={events} />
    </div>
  );
}
