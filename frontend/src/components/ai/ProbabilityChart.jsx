export default function ProbabilityChart({ probabilities = {} }) {
  return <section><h3>Class Probabilities</h3>{Object.entries(probabilities).map(([label, value]) => <p key={label}>{label}: {(value * 100).toFixed(2)}%</p>)}</section>;
}
