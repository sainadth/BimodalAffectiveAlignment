import { useRef, useState } from 'react'

type Props = {
  name: string
  label: string
  low: string
  high: string
  value: number | null
  onChange: (v: number) => void
}

/**
 * Continuous 0–100 judgement. No numeric readout and no handle until the person
 * touches it: a visible number invites round answers, and a pre-placed handle
 * anchors everyone to the middle. Both would flatten exactly the differences the
 * study is trying to detect.
 */
export default function Scale({ name, label, low, high, value, onChange }: Props) {
  const [touched, setTouched] = useState(value !== null)
  const ref = useRef<HTMLInputElement>(null)

  const commit = (v: number) => {
    setTouched(true)
    onChange(v)
  }

  const position = value ?? 50
  const spoken =
    value === null
      ? 'not answered yet'
      : value <= 12 ? low
      : value >= 88 ? high
      : value < 44 ? `closer to ${low}`
      : value > 56 ? `closer to ${high}`
      : 'in the middle'

  return (
    <div className="scale" data-unset={!touched}>
      <p id={`${name}-label`}>{label}</p>
      <input
        ref={ref}
        type="range"
        min={0}
        max={100}
        step={1}
        value={position}
        aria-labelledby={`${name}-label`}
        aria-valuetext={spoken}
        onChange={(e) => commit(Number(e.target.value))}
        onPointerDown={() => setTouched(true)}
      />
      <div className="anchors">
        <span>{low}</span>
        <span>{high}</span>
      </div>
    </div>
  )
}
