import { useEffect, useRef, useState } from 'react'

export type CameraHandle = { capture: (n?: number) => Promise<string[]> }

type Props = { onReady: (h: CameraHandle | null) => void }

/** Live preview plus a short burst taken on Send. Frames are analysed and dropped. */
export default function Camera({ onReady }: Props) {
  const video = useRef<HTMLVideoElement>(null)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    let stream: MediaStream | null = null
    navigator.mediaDevices
      .getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' }, audio: false })
      .then((s) => {
        stream = s
        if (video.current) video.current.srcObject = s
        onReady({
          capture: async (n = 3) => {
            const out: string[] = []
            const v = video.current
            if (!v || v.videoWidth === 0) return out
            const c = document.createElement('canvas')
            c.width = v.videoWidth
            c.height = v.videoHeight
            const ctx = c.getContext('2d')!
            for (let i = 0; i < n; i++) {
              ctx.drawImage(v, 0, 0)
              out.push(c.toDataURL('image/jpeg', 0.85))
              if (i < n - 1) await new Promise((r) => setTimeout(r, 250))
            }
            return out
          },
        })
      })
      .catch((e) => {
        setErr(`Camera unavailable: ${e.message}. Allow camera access, then reload this page.`)
        onReady(null)
      })
    return () => {
      stream?.getTracks().forEach((t) => t.stop())
      onReady(null)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  if (err) return <p className="notice">{err}</p>

  return (
    <div className="lens">
      <video ref={video} autoPlay playsInline muted aria-label="Camera preview" />
      <p>A few frames are taken when you send, then analysed and discarded.</p>
    </div>
  )
}
