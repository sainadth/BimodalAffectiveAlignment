import { useEffect, useState } from 'react'
import { api, type SessionInfo } from './api'
import Camera, { type CameraHandle } from './Camera'
import Choice from './Choice'
import Dictate from './Dictate'
import Markdown from './Markdown'
import Scale from './Scale'

type Phase = 'loading' | 'error' | 'consent' | 'trial' | 'pets' | 'debrief' | 'done'
type Beat = 'write' | 'sending' | 'feel' | 'rate'

function tokenFromPath(): string | null {
  const m = window.location.pathname.match(/^\/p\/([A-Za-z0-9_-]+)/)
  return m ? m[1] : null
}

function TurnHead({ done, total, trailing }: { done: number; total: number; trailing?: string }) {
  return (
    <div className="turnhead">
      <span>
        Turn {done + 1} of {total}
      </span>
      <i style={{ ['--done' as string]: `${(done / total) * 100}%` }} />
      {trailing && <span>{trailing}</span>}
    </div>
  )
}

export default function App() {
  const token = tokenFromPath()
  const [phase, setPhase] = useState<Phase>('loading')
  const [info, setInfo] = useState<SessionInfo | null>(null)
  const [error, setError] = useState('')
  const [trial, setTrial] = useState(0)

  useEffect(() => {
    if (!token) {
      setError('This page needs the personal link from the researcher.')
      setPhase('error')
      return
    }
    api
      .session(token)
      .then((s) => {
        setInfo(s)
        setTrial(s.trials_done)
        if (s.trials_done >= s.trials_total) setPhase('pets')
        else setPhase(s.consented ? 'trial' : 'consent')
      })
      .catch((e) => {
        setError(e.message)
        setPhase('error')
      })
  }, [token])

  if (phase === 'loading') return <main><p className="hint">Loading…</p></main>
  if (phase === 'error' || !info || !token) return <main><p className="notice">{error}</p></main>

  return (
    <main>
      {phase === 'consent' && (
        <Consent
          info={info}
          onAgree={async () => {
            await api.consent(token, info.consent_version)
            setPhase('trial')
          }}
        />
      )}
      {phase === 'trial' && (
        <Turn
          key={trial}
          token={token}
          info={info}
          trial={trial}
          onDone={(finished) => (finished ? setPhase('pets') : setTrial((t) => t + 1))}
        />
      )}
      {phase === 'pets' && <Closing token={token} info={info} onDone={() => setPhase('debrief')} />}
      {phase === 'debrief' && <Debrief token={token} info={info} onDone={() => setPhase('done')} />}
      {phase === 'done' && (
        <div className="closing">
          <p>That is everything. Thank you for your time — you can close this window.</p>
        </div>
      )}
    </main>
  )
}

function Consent({ info, onAgree }: { info: SessionInfo; onAgree: () => Promise<void> }) {
  const [agree, setAgree] = useState(false)
  const [busy, setBusy] = useState(false)
  return (
    <div>
      <div className="prose">
        <Markdown text={info.consent_md} />
      </div>
      <label className="agree">
        <input type="checkbox" checked={agree} onChange={(e) => setAgree(e.target.checked)} />
        <span>I have read this and agree to take part.</span>
      </label>
      <div className="actions">
        <button
          className="primary"
          disabled={!agree || busy}
          onClick={async () => {
            setBusy(true)
            await onAgree()
          }}
        >
          Start
        </button>
      </div>
    </div>
  )
}

function Turn({
  token,
  info,
  trial,
  onDone,
}: {
  token: string
  info: SessionInfo
  trial: number
  onDone: (finished: boolean) => void
}) {
  const [beat, setBeat] = useState<Beat>('write')
  const [text, setText] = useState('')
  const [dictated, setDictated] = useState('')
  const [cam, setCam] = useState<CameraHandle | null>(null)
  const [felt, setFelt] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [reply, setReply] = useState('')
  const [manip, setManip] = useState<string | null>(null)
  const [ratings, setRatings] = useState<Record<string, number | null>>(
    Object.fromEntries(info.trial_items.map((i) => [i.key, null])),
  )
  const [err, setErr] = useState('')
  const scripted = info.scripted[String(trial)]

  const send = async () => {
    setErr('')
    setBeat('sending')
    try {
      const frames = cam ? await cam.capture(3) : []
      if (frames.length === 0) throw new Error('No camera frame was captured. Allow camera access and try again.')
      const method = !dictated ? 'typed' : text.trim() === dictated.trim() ? 'spoken' : 'spoken+edited'
      await api.analyze(token, trial, text.trim(), frames, method)
      setBeat('feel')
    } catch (e) {
      setErr((e as Error).message)
      setBeat('write')
    }
  }

  const reveal = async () => {
    if (!felt) return
    setErr('')
    setBusy(true)
    try {
      const r = await api.felt(token, trial, felt)
      setReply(r.reply)
      setBeat('rate')
    } catch (e) {
      setErr((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const save = async () => {
    setErr('')
    setBusy(true)
    try {
      const r = await api.rate(
        token,
        trial,
        Object.fromEntries(Object.entries(ratings).map(([k, v]) => [k, v as number])),
        manip!,
      )
      onDone(r.finished)
    } catch (e) {
      setErr((e as Error).message)
      setBusy(false)
    }
  }

  const canSend = text.trim().length > 0 && cam !== null
  const canSave = Object.values(ratings).every((v) => v !== null) && manip !== null

  return (
    <div>
      <TurnHead done={trial} total={info.trials_total} />

      {(beat === 'write' || beat === 'sending') && (
        <div>
          {scripted ? (
            <p className="instruction">{scripted}</p>
          ) : (
            <p className="lede">Write a few lines about how you are feeling right now, or about your day.</p>
          )}
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Type here…"
            disabled={beat === 'sending'}
            aria-label="Your message"
          />
          {info.dictation && (
            <Dictate
              token={token}
              disabled={beat === 'sending'}
              onText={(v) => {
                const next = text.trim() ? `${text.trim()} ${v}` : v
                setText(next)
                setDictated(next)
              }}
            />
          )}
          <div className="capture">
            <Camera onReady={setCam} />
            <div className="sendbar">
              <button className="primary" disabled={!canSend || beat === 'sending'} onClick={send}>
                {beat === 'sending' ? 'Sending…' : 'Send'}
              </button>
              {!canSend && beat === 'write' && (
                <span className="hint">A message and camera access are both needed.</span>
              )}
            </div>
          </div>
          {err && <p className="notice">{err}</p>}
        </div>
      )}

      {beat === 'feel' && (
        <div className="feel">
          <p>Before you see the reply</p>
          <h2>Right now, I mostly feel…</h2>
          <Choice
            name={`felt-${trial}`}
            options={info.felt_options}
            value={felt}
            onChange={setFelt}
            legend="Right now I mostly feel"
          />
          <div className="actions" style={{ justifyContent: 'center' }}>
            <button className="primary" disabled={!felt || busy} onClick={reveal}>
              {busy ? 'One moment…' : 'Show the reply'}
            </button>
          </div>
          {err && <p className="notice">{err}</p>}
        </div>
      )}

      {beat === 'rate' && (
        <div>
          <p className="echo">{text}</p>
          <div className="reply">{reply}</div>

          {info.trial_items.map((it) => (
            <Scale
              key={it.key}
              name={`${it.key}-${trial}`}
              label={it.text}
              low={info.scale.low}
              high={info.scale.high}
              value={ratings[it.key]}
              onChange={(v) => setRatings({ ...ratings, [it.key]: v })}
            />
          ))}

          <hr className="divider" />
          <p className="lede">{info.manip_check_text}</p>
          <Choice
            name={`manip-${trial}`}
            options={info.manip_options}
            value={manip}
            onChange={setManip}
            legend={info.manip_check_text}
          />

          <div className="actions">
            <button className="primary" disabled={!canSave || busy} onClick={save}>
              {busy ? 'Saving…' : 'Continue'}
            </button>
            {!canSave && <span className="hint">Every question needs an answer.</span>}
          </div>
          {err && <p className="notice">{err}</p>}
        </div>
      )}
    </div>
  )
}

function Closing({ token, info, onDone }: { token: string; info: SessionInfo; onDone: () => void }) {
  const [vals, setVals] = useState<Record<string, number | null>>(
    Object.fromEntries(info.pets_items.map((i) => [i.key, null])),
  )
  const [funnel, setFunnel] = useState<string | null>(null)
  const [free, setFree] = useState('')
  const [err, setErr] = useState('')
  const [busy, setBusy] = useState(false)
  const ready = Object.values(vals).every((v) => v !== null) && funnel !== null

  return (
    <div>
      <TurnHead done={info.trials_total} total={info.trials_total} trailing="last part" />
      <p className="lede">
        Thinking back over the whole conversation, how much do you agree with each of these?
      </p>
      {info.pets_items.map((it) => (
        <Scale
          key={it.key}
          name={it.key}
          label={it.text}
          low={info.scale.low}
          high={info.scale.high}
          value={vals[it.key]}
          onChange={(v) => setVals({ ...vals, [it.key]: v })}
        />
      ))}

      <hr className="divider" />
      <p className="lede">Did you notice any pattern in how the agent replied?</p>
      <Choice name="funnel" options={['No', 'Unsure', 'Yes']} value={funnel} onChange={setFunnel} legend="Did you notice any pattern?" />
      <label className="field" htmlFor="funnel-text">
        If you did, what was it? (optional)
      </label>
      <input id="funnel-text" type="text" value={free} onChange={(e) => setFree(e.target.value)} style={{ maxWidth: '100%' }} />

      <div className="actions">
        <button
          className="primary"
          disabled={!ready || busy}
          onClick={async () => {
            setBusy(true)
            try {
              await api.pets(
                token,
                Object.fromEntries(Object.entries(vals).map(([k, v]) => [k, v as number])),
                funnel!.toLowerCase(),
                free,
              )
              onDone()
            } catch (e) {
              setErr((e as Error).message)
              setBusy(false)
            }
          }}
        >
          {busy ? 'Saving…' : 'Finish'}
        </button>
      </div>
      {err && <p className="notice">{err}</p>}
    </div>
  )
}

function Debrief({ token, info, onDone }: { token: string; info: SessionInfo; onDone: () => void }) {
  const [removed, setRemoved] = useState<number | null>(null)
  const [busy, setBusy] = useState(false)
  if (removed !== null)
    return (
      <div className="closing">
        <p>Your data are deleted — {removed} records removed. Thank you for your time.</p>
      </div>
    )
  return (
    <div>
      <div className="prose">
        <Markdown text={info.debrief_md} />
      </div>
      <div className="actions section-gap">
        <button className="primary" onClick={onDone}>
          Keep my data
        </button>
        <button
          disabled={busy}
          onClick={async () => {
            setBusy(true)
            const r = await api.withdraw(token)
            setRemoved(r.rows_removed)
          }}
        >
          Delete my data
        </button>
      </div>
    </div>
  )
}
