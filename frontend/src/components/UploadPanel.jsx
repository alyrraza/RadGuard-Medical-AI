import { useRef, useState } from 'react'

export default function UploadPanel({
  preview, report, loading, error,
  onImageSelect, onReportChange, onSubmit,
}) {
  const inputRef = useRef()
  const [dragging, setDragging] = useState(false)

  function handleDrop(e) {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith('image/')) onImageSelect(file)
  }

  function handleFileChange(e) {
    const file = e.target.files[0]
    if (file) onImageSelect(file)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

      {/* Drop zone */}
      <div
        onClick={() => inputRef.current.click()}
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        style={{
          border: `2px dashed ${dragging ? 'var(--accent)' : 'var(--border)'}`,
          borderRadius: 'var(--radius)',
          background: dragging ? 'var(--accent-dim)' : 'var(--surface)',
          cursor: 'pointer',
          minHeight: 220,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 12,
          transition: 'border-color 0.15s, background 0.15s',
          overflow: 'hidden',
          position: 'relative',
        }}
      >
        {preview ? (
          <img
            src={preview}
            alt="X-ray preview"
            style={{
              maxHeight: 280,
              maxWidth: '100%',
              borderRadius: 8,
              objectFit: 'contain',
            }}
          />
        ) : (
          <>
            <span style={{ fontSize: 40, opacity: 0.5 }}>🩻</span>
            <span style={{ color: 'var(--muted)', textAlign: 'center', lineHeight: 1.5 }}>
              Drop chest X-ray here<br />
              <span style={{ fontSize: 12 }}>or click to browse — JPEG / PNG</span>
            </span>
          </>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          style={{ display: 'none' }}
          onChange={handleFileChange}
        />
      </div>

      {/* Report textarea */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <label style={{ fontSize: 13, fontWeight: 600, color: 'var(--muted)' }}>
          AI-Generated Report Text
        </label>
        <textarea
          value={report}
          onChange={e => onReportChange(e.target.value)}
          placeholder="Paste the AI-generated radiology report here…&#10;&#10;e.g. The heart is mildly enlarged. No pleural effusion. Lung fields are clear."
          rows={6}
          style={{
            background: 'var(--surface)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius)',
            color: 'var(--text)',
            padding: '10px 12px',
            fontSize: 13,
            fontFamily: 'inherit',
            resize: 'vertical',
            width: '100%',
            lineHeight: 1.6,
          }}
        />
      </div>

      {/* Error banner */}
      {error && (
        <div style={{
          background: '#2d1216',
          border: '1px solid #6e1b1b',
          borderRadius: 'var(--radius)',
          padding: '10px 14px',
          color: 'var(--red)',
          fontSize: 13,
        }}>
          {error}
        </div>
      )}

      {/* Submit */}
      <button
        onClick={onSubmit}
        disabled={loading}
        style={{
          background: loading ? 'var(--accent-dim)' : 'var(--accent)',
          color: '#fff',
          border: 'none',
          borderRadius: 'var(--radius)',
          padding: '12px 24px',
          fontSize: 14,
          fontWeight: 600,
          cursor: loading ? 'not-allowed' : 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 8,
          transition: 'background 0.15s',
          width: '100%',
        }}
      >
        {loading ? (
          <>
            <Spinner />
            Analyzing…
          </>
        ) : (
          'Analyze Report'
        )}
      </button>
    </div>
  )
}

function Spinner() {
  return (
    <span style={{
      width: 14, height: 14,
      border: '2px solid rgba(255,255,255,0.3)',
      borderTopColor: '#fff',
      borderRadius: '50%',
      display: 'inline-block',
      animation: 'spin 0.7s linear infinite',
    }} />
  )
}
