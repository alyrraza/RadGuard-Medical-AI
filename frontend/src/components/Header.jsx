export default function Header() {
  return (
    <header style={{
      borderBottom: '1px solid var(--border)',
      background: 'var(--surface)',
      padding: '0 24px',
    }}>
      <div style={{
        maxWidth: 1100,
        margin: '0 auto',
        height: 56,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: 20 }}>🫁</span>
          <span style={{ fontWeight: 700, fontSize: 16, letterSpacing: '-0.3px' }}>
            RadGuard
          </span>
          <span style={{
            fontSize: 11,
            fontWeight: 500,
            color: 'var(--muted)',
            background: 'var(--bg)',
            border: '1px solid var(--border)',
            borderRadius: 4,
            padding: '2px 7px',
            marginLeft: 4,
          }}>
            V11
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <span style={{ color: 'var(--muted)', fontSize: 13 }}>
            AI Radiology Report Error Detection
          </span>
          <a
            href="/health"
            target="_blank"
            rel="noreferrer"
            style={{
              fontSize: 12,
              color: 'var(--green)',
              textDecoration: 'none',
              display: 'flex',
              alignItems: 'center',
              gap: 5,
            }}
          >
            <span style={{
              width: 7, height: 7,
              borderRadius: '50%',
              background: 'var(--green)',
              display: 'inline-block',
            }} />
            API
          </a>
        </div>
      </div>
    </header>
  )
}
