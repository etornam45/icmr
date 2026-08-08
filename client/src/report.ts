import type { EventItem } from "./api";

export function buildReportText(events: EventItem[]): string {
  const ordered = [...events].reverse()
  const generated = new Date()

  const lines: string[] = []
  const rule = '='.repeat(64)


  lines.push(rule)
  lines.push('ICMR - ANOMALY EVENT REPORT')
  lines.push(rule)
  lines.push(`Generated:    ${generated.toLocaleString()}`)
  lines.push(`Total events: ${events.length}`)

  if (ordered.length > 0) {
    lines.push(`Window:       ${ordered[0].iso_time}  ->  ${ordered[ordered.length - 1].iso_time}`)
    const byClass = new Map<string, number>()
    let peak = ordered[0]
    for (const ev of ordered) {
      byClass.set(ev.anomaly_class, (byClass.get(ev.anomaly_class) ?? 0) + 1)
      if (ev.score > peak.score) peak = ev
    }

    lines.push(`Peak score:   ${peak.score.toFixed(3)} (${peak.anomaly_class})`)
    lines.push('')
    lines.push('BREAKDOWN BY CLASS')
    lines.push('-'.repeat(64))
    for (const [cls, n] of [...byClass].sort((a, b) => b[1] - a[1])) {
      lines.push(`  ${cls.padEnd(28)} ${String(n).padStart(4)}`)
    }
  }

  lines.push('')
  lines.push('EVENT LOG')
  lines.push('-'.repeat(64))

  if (ordered.length === 0) {
    lines.push('  (no events recorded)')
  } else {
    ordered.forEach((ev, i) => {
      lines.push(`[${String(i + 1).padStart(3, '0')}] ${ev.iso_time}`)
      lines.push(`      Class:   ${ev.anomaly_class}`)
      lines.push(`      Score:   ${ev.score.toFixed(3)}`)
      lines.push(`      Caption: ${ev.caption ?? '(none)'}`)
      lines.push('')
    })
  }

  lines.push(rule)
  lines.push('Scores are model confidence values and are not a substitute')
  lines.push('for human review of the source footage.')
  lines.push(rule)

  return lines.join('\r\n')
}


export function downloadReport(events: EventItem[]) {
  const text = buildReportText(events)
  const stamp = new Date().toISOString().replace(/[:T]/g, '-').slice(0, 19)

  const blob = new Blob([text], { type: 'text/plain;charset=utf-8'})
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `icmr-report-${stamp}.txt`
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}