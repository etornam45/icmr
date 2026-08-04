import type { OverlayMode } from '../api'

type Props = {
  mode: OverlayMode
  onChange: (mode: OverlayMode) => void
  disabled?: boolean
}

const MODES: { id: OverlayMode; label: string }[] = [
  { id: 'none', label: 'None' },
  { id: 'detection', label: 'Detection' },
  { id: 'pca', label: 'DINO PCA' },
]

export function OverlayToggle({ mode, onChange, disabled }: Props) {
  return (
    <div className="mode-field">
      <span className="field-label" id="overlay-mode-label">
        Mode
      </span>
      <div
        className="overlay-toggle"
        role="group"
        aria-labelledby="overlay-mode-label"
      >
        {MODES.map((m) => (
          <button
            key={m.id}
            type="button"
            className={mode === m.id ? 'active' : ''}
            disabled={disabled}
            onClick={() => onChange(m.id)}
          >
            {m.label}
          </button>
        ))}
      </div>
    </div>
  )
}
