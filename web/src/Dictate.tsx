import { useRef, useState } from 'react'
import { api } from './api'

type Props = {
  token: string
  disabled?: boolean
  onText: (text: string) => void
}

type State = 'idle' | 'recording' | 'working'

/**
 * Optional dictation. The microphone is only requested if the person chooses to
 * speak, and the transcript lands in their own text box so they can fix it before
 * sending. Audio is dropped as soon as it is transcribed.
 */
export default function Dictate({ token, disabled, onText }: Props) {
  const [state, setState] = useState<State>('idle')
  const [err, setErr] = useState('')
  const rec = useRef<MediaRecorder | null>(null)
  const chunks = useRef<Blob[]>([])

  const stop = () => {
    rec.current?.stop()
    rec.current?.stream.getTracks().forEach((t) => t.stop())
    rec.current = null
  }

  const start = async () => {
    setErr('')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mr = new MediaRecorder(stream)
      chunks.current = []
      mr.ondataavailable = (e) => e.data.size && chunks.current.push(e.data)
      mr.onstop = async () => {
        setState('working')
        try {
          const blob = new Blob(chunks.current, { type: mr.mimeType || 'audio/webm' })
          if (blob.size < 1200) throw new Error('That recording was too short.')
          const b64: string = await new Promise((res, rej) => {
            const r = new FileReader()
            r.onload = () => res(String(r.result))
            r.onerror = () => rej(new Error('Could not read the recording.'))
            r.readAsDataURL(blob)
          })
          const { text } = await api.transcribe(token, b64)
          if (!text) throw new Error('Nothing was picked up. Try again, or type instead.')
          onText(text)
        } catch (e) {
          setErr((e as Error).message)
        } finally {
          setState('idle')
        }
      }
      mr.start()
      rec.current = mr
      setState('recording')
    } catch {
      setErr('Microphone unavailable. Type your message instead.')
      setState('idle')
    }
  }

  return (
    <div className="dictate">
      <button
        type="button"
        className={state === 'recording' ? 'recording' : ''}
        disabled={disabled || state === 'working'}
        onClick={state === 'recording' ? stop : start}
        aria-pressed={state === 'recording'}
      >
        {state === 'recording' ? 'Stop and transcribe' : state === 'working' ? 'Transcribing…' : 'Speak instead'}
      </button>
      {state === 'recording' && <span className="hint">Recording. Speak, then press stop.</span>}
      {state === 'idle' && !err && <span className="hint">You can edit the words afterwards.</span>}
      {err && <span className="notice">{err}</span>}
    </div>
  )
}
