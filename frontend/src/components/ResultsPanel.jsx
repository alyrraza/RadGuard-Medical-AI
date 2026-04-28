import ConditionCard from './ConditionCard'

const GRADE_COLOR = {
  Excellent: '#3fb950',
  Good:      '#8bc34a',
  Fair:      '#d29922',
  Poor:      '#f85149',
  Critical:  '#bc8cff',
}

export default function ResultsPanel({ result, preview }) {
  const {
    task1_elrrs: elrrs,
    task1_conditions: conditions = [],
    task2_xray_findings: xrayFindings = {},
    task3_heatmaps: heatmaps = {},
    not_mentioned = [],
    sentences_analyzed = 0,
  } = result

  const gradeColor = elrrs ? (GRADE_COLOR[elrrs.grade] || '#7d8590') : '#7d8590'
  const activeHeatmap = Object.entries(heatmaps)[0]

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

      {/* ELRRs Score Card */}
      {elrrs && (
        <div style={{
          background: 'var(--surface)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius)',
          padding: '16px 20px',
          display: 'flex',
          alignItems: 'center',
          gap: 20,
        }}>
          {/* Score circle */}
          <div style={{
            width: 72, height: 72,
            borderRadius: '50%',
            border: `3px solid ${gradeColor}`,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
          }}>
            <span style={{ fontSize: 20, fontWeight: 700, color: gradeColor, lineHeight: 1 }}>
              {elrrs.score}
            </span>
            <span style={{ fontSize: 9, color: 'var(--muted)', marginTop: 1 }}>ELRRs</span>
          </div>

          {/* Grade info */}
          <div style={{ flex: 1 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
              <span style={{ fontWeight: 700, fontSize: 16, color: gradeColor }}>
                {elrrs.grade}
              </span>
              <span style={{ fontSize: 12, color: 'var(--muted)' }}>
                {elrrs.grade_desc}
              </span>
            </div>
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
              {[
                { label: 'Supported',    val: elrrs.supported_count,    color: '#3fb950' },
                { label: 'Hallucinated', val: elrrs.hallucinated_count, color: '#f85149' },
                { label: 'Missing',      val: elrrs.missing_count,      color: '#d29922' },
                { label: 'Inaccurate',   val: elrrs.inaccurate_count,   color: '#bc8cff' },
              ].map(({ label, val, color }) => (
                <span key={label} style={{ fontSize: 12, color }}>
                  {val} {label}
                </span>
              ))}
            </div>
            <div style={{ marginTop: 4, fontSize: 11, color: 'var(--muted)' }}>
              {sentences_analyzed} sentence{sentences_analyzed !== 1 ? 's' : ''} analyzed
              · {not_mentioned.length} conditions not mentioned
            </div>
          </div>
        </div>
      )}

      {/* Heatmap (first available) */}
      {activeHeatmap && (
        <div style={{
          background: 'var(--surface)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius)',
          overflow: 'hidden',
        }}>
          <div style={{
            padding: '10px 14px',
            borderBottom: '1px solid var(--border)',
            fontSize: 12,
            fontWeight: 600,
            color: 'var(--muted)',
            display: 'flex',
            justifyContent: 'space-between',
          }}>
            <span>Attention Heatmap — {activeHeatmap[0].replace(/_/g, ' ')}</span>
            <a
              href={activeHeatmap[1]}
              target="_blank"
              rel="noreferrer"
              style={{ color: 'var(--accent)', textDecoration: 'none', fontSize: 11 }}
            >
              Open full ↗
            </a>
          </div>
          <img
            src={activeHeatmap[1]}
            alt={`Attention map for ${activeHeatmap[0]}`}
            style={{ width: '100%', display: 'block', maxHeight: 280, objectFit: 'contain' }}
          />
          {Object.keys(heatmaps).length > 1 && (
            <div style={{
              padding: '8px 14px',
              borderTop: '1px solid var(--border)',
              display: 'flex',
              gap: 6,
              flexWrap: 'wrap',
            }}>
              {Object.entries(heatmaps).map(([cond, url]) => (
                <a
                  key={cond}
                  href={url}
                  target="_blank"
                  rel="noreferrer"
                  style={{
                    fontSize: 11,
                    color: 'var(--accent)',
                    textDecoration: 'none',
                    background: 'var(--bg)',
                    border: '1px solid var(--border)',
                    borderRadius: 4,
                    padding: '2px 7px',
                  }}
                >
                  {cond.replace(/_/g, ' ')}
                </a>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Condition cards */}
      {conditions.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <h3 style={{ fontSize: 13, fontWeight: 600, color: 'var(--muted)', margin: 0 }}>
            Active Conditions ({conditions.length})
          </h3>
          {conditions.map(cond => (
            <ConditionCard key={cond.name} condition={cond} />
          ))}
        </div>
      )}

      {/* Not mentioned */}
      {not_mentioned.length > 0 && (
        <details style={{ cursor: 'pointer' }}>
          <summary style={{
            fontSize: 12,
            color: 'var(--muted)',
            padding: '6px 0',
            userSelect: 'none',
          }}>
            {not_mentioned.length} conditions not mentioned in report
          </summary>
          <div style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: 6,
            marginTop: 8,
          }}>
            {not_mentioned.map(c => (
              <span key={c} style={{
                fontSize: 11,
                color: 'var(--muted)',
                background: 'var(--surface)',
                border: '1px solid var(--border)',
                borderRadius: 4,
                padding: '2px 8px',
              }}>
                {c.replace(/_/g, ' ')}
              </span>
            ))}
          </div>
        </details>
      )}
    </div>
  )
}
