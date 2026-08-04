import { useRef, useState } from 'react'
import type { SourceType } from '../api'

type Props = {
  busy: boolean
  running: boolean
  onStart: (type: SourceType, uri: string) => Promise<void>
  onUpload: (file: File) => Promise<void>
  onStop: () => Promise<void>
}

export function SourceControls({
  busy,
  running,
  onStart,
  onUpload,
  onStop,
}: Props) {
  const [type, setType] = useState<SourceType>('file')
  const [uri, setUri] = useState('')
  const [fileName, setFileName] = useState<string | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const canStart =
    type === 'stream' ? uri.trim().length > 0 : file != null

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    try {
      if (type === 'file') {
        if (!file) throw new Error('Choose a video file to upload')
        await onUpload(file)
      } else {
        await onStart('stream', uri.trim())
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  return (
    <form className="source-controls" onSubmit={submit}>
      <div className="source-row">
        <label>
          Source
          <select
            value={type}
            onChange={(e) => {
              setType(e.target.value as SourceType)
              setError(null)
            }}
            disabled={busy || running}
          >
            <option value="file">Upload video</option>
            <option value="stream">Stream URL</option>
          </select>
        </label>

        {type === 'file' ? (
          <div className="uri-field upload-field">
            <span className="field-label">Video</span>
            <div className="upload-row">
              <input
                ref={inputRef}
                type="file"
                accept="video/*,.mp4,.avi,.mov,.mkv,.webm,.mpg,.mpeg,.m4v"
                disabled={busy || running}
                onChange={(e) => {
                  const next = e.target.files?.[0] ?? null
                  setFile(next)
                  setFileName(next?.name ?? null)
                }}
              />
              <button
                type="button"
                className="browse"
                disabled={busy || running}
                onClick={() => inputRef.current?.click()}
              >
                Browse
              </button>
              <span className="file-name" title={fileName ?? undefined}>
                {fileName ?? 'No file selected'}
              </span>
            </div>
          </div>
        ) : (
          <label className="uri-field">
            URL
            <input
              value={uri}
              onChange={(e) => setUri(e.target.value)}
              placeholder="rtsp://… or https://….m3u8"
              disabled={busy || running}
            />
          </label>
        )}

        {!running ? (
          <button type="submit" disabled={busy || !canStart}>
            {type === 'file' ? 'Upload & Start' : 'Start'}
          </button>
        ) : (
          <button
            type="button"
            className="stop"
            disabled={busy}
            onClick={async () => {
              setError(null)
              try {
                await onStop()
              } catch (err) {
                setError(err instanceof Error ? err.message : String(err))
              }
            }}
          >
            Stop
          </button>
        )}
      </div>
      {error && <p className="form-error">{error}</p>}
    </form>
  )
}
