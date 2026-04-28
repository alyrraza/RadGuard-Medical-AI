const VERDICT_STYLE = {
  SUPPORTED:     { color: '#3fb950', bg: '#0d2318', label: '✓ Supported' },
  HALLUCINATED:  { color: '#f85149', bg: '#2d1216', label: '✗ Hallucinated' },
  MISSING:       { color: '#d29922', bg: '#2a1f00', label: '⚠ Missing' },
  INACCURATE:    { color: '#bc8cff', bg: '#1e1535', label: '~ Inaccurate' },
  NOT_MENTIONED: { color: '#7d8590', bg: 'transparent', label: '– Not mentioned' },
}

export default function ConditionCard({ condition }) {
  const { name, verdict, confidence, meaning, source_text, xray_present } = condition
  const style = VERDICT_STYLE[verdict] || VERDICT_STYLE.NOT_MENTIONED
  const confPct = Math.round((confidence || 0) * 100)

  return (
    <div style={{
      background: 'var(--surface)',
      border: `1px solid var(--border)`,
      borderLeft: `3px solid ${style.color}`,
      borderRadius: 'var(--radius)',
      padding: '12px 14px',
      display: 'flex',
      flexDirection: 'column',
      gap: 6,
    }}>
      {/* Top row */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontWeight: 600, fontSize: 13 }}>
          {name.replace(/_/g, ' ')}
        </span>
        <span style={{
          fontSize: 11,
          fontWeight: 600,
          color: style.color,
          background: style.bg,
          border: `1px solid ${style.color}33`,
          borderRadius: 4,
          padding: '2px 8px',
        }}>
          {style.label}
        </span>
      </div>

      {/* Confidence bar */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <div style={{
          flex: 1, height: 4,
          background: 'var(--border)',
          borderRadius: 2,
          overflow: 'hidden',
        }}>
          <div style={{
            width: `${confPct}%`,
            height: '100%',
            background: style.color,
            borderRadius: 2,
            transition: 'width 0.4s ease',
          }} />
        </div>
        <span style={{ fontSize: 11, color: 'var(--muted)', minWidth: 32, textAlign: 'right' }}>
          {confPct}%
        </span>
      </div>

      {/* Meaning */}
      {meaning && (
        <span style={{ fontSize: 12, color: 'var(--muted)', lineHeight: 1.4 }}>
          {meaning}
        </span>
      )}

      {/* Source sentence */}
      {source_text && (
        <span style={{
          fontSize: 11,
          color: 'var(--muted)',
          fontStyle: 'italic',
          borderLeft: '2px solid var(--border)',
          paddingLeft: 8,
          marginTop: 2,
          lineHeight: 1.4,
        }}>
          "{source_text.length > 120 ? source_text.slice(0, 120) + '…' : source_text}"
        </span>
      )}

      {/* X-ray finding pill */}
      <div style={{ display: 'flex', gap: 6, marginTop: 2 }}>
        <span style={{
          fontSize: 10,
          color: xray_present ? '#3fb950' : '#7d8590',
          background: 'var(--bg)',
          border: '1px solid var(--border)',
          borderRadius: 3,
          padding: '1px 6px',
        }}>
          X-ray: {xray_present ? 'present' : 'absent'}
        </span>
      </div>
    </div>
  )
}
