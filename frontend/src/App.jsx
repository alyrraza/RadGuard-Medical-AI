import { useState } from 'react'
import Header from './components/Header'
import UploadPanel from './components/UploadPanel'
import ResultsPanel from './components/ResultsPanel'

const API_URL = import.meta.env.VITE_API_URL || ''

export default function App() {
  const [image, setImage]       = useState(null)   // File object
  const [preview, setPreview]   = useState(null)   // object URL
  const [report, setReport]     = useState('')
  const [loading, setLoading]   = useState(false)
  const [result, setResult]     = useState(null)
  const [error, setError]       = useState(null)

  function handleImageSelect(file) {
    setImage(file)
    setPreview(URL.createObjectURL(file))
    setResult(null)
    setError(null)
  }

  async function handleSubmit() {
    if (!image)  { setError('Please upload a chest X-ray image.'); return }
    if (!report.trim()) { setError('Please enter the AI report text.'); return }

    setLoading(true)
    setError(null)
    setResult(null)

    const form = new FormData()
    form.append('file', image)
    form.append('ai_report', report.trim())

    try {
      const res = await fetch(`${API_URL}/analyze`, { method: 'POST', body: form })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.error || `Server error ${res.status}`)
      }
      const data = await res.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Header />

      <main style={{
        flex: 1,
        maxWidth: 1100,
        width: '100%',
        margin: '0 auto',
        padding: '32px 16px',
        display: 'grid',
        gridTemplateColumns: result ? '1fr 1fr' : '1fr',
        gap: 24,
        alignItems: 'start',
      }}>
        <UploadPanel
          preview={preview}
          report={report}
          loading={loading}
          error={error}
          onImageSelect={handleImageSelect}
          onReportChange={setReport}
          onSubmit={handleSubmit}
        />

        {result && <ResultsPanel result={result} preview={preview} />}
      </main>
    </div>
  )
}
