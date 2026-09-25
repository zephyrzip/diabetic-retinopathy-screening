export default function DRResultCard({ severity, level, confidence }) {
  return <section><h2>{severity}</h2><p>Level {level}</p><p>Confidence: {(confidence * 100).toFixed(2)}%</p></section>;
}
