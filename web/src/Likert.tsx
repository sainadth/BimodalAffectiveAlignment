type Props = { name: string; label: string; value: number | null; onChange: (v: number) => void }

export default function Likert({ name, label, value, onChange }: Props) {
  return (
    <div>
      <div>{label}</div>
      <div className="likert" role="radiogroup" aria-label={label}>
        {[1, 2, 3, 4, 5].map((v) => (
          <label key={v}>
            <input type="radio" name={name} checked={value === v} onChange={() => onChange(v)} />
            <span>{v}</span>
          </label>
        ))}
      </div>
    </div>
  )
}
