type Props = {
  name: string
  options: string[]
  value: string | null
  onChange: (v: string) => void
  legend?: string
}

/** Emotions are words here, so they are set as words — no pills, no colour coding. */
export default function Choice({ name, options, value, onChange, legend }: Props) {
  return (
    <div className="words" role="radiogroup" aria-label={legend ?? name}>
      {options.map((o) => (
        <label key={o}>
          <input
            type="radio"
            name={name}
            value={o}
            checked={value === o}
            onChange={() => onChange(o)}
          />
          <span>{o}</span>
        </label>
      ))}
    </div>
  )
}
