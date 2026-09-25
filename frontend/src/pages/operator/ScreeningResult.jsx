import { mockResult } from "../../data/mockResults";

export default function ScreeningResult() {
  return <main><h1>AI Screening Result</h1><h2>{mockResult.severity}</h2><p>Level: {mockResult.predictedLevel}</p><p>Confidence: {(mockResult.confidence * 100).toFixed(2)}%</p><p>Referable: {mockResult.referable ? "Yes" : "No"}</p></main>;
}
