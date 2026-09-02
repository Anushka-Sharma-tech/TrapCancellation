import { RiskResult } from "../lib/riskScorer";

const labels = {
  acousticArtifact: "Acoustic Artifact",
  prosodyAnomaly: "Prosody Anomaly",
  speakerDrift: "Speaker Drift",
  behavioralRisk: "Behavioral Risk"
};

export default function AlertPanel({ result }: { result: RiskResult }) {
  return (
    <section className="panel">
      <h2>Explainability Signals</h2>

      {Object.entries(result.signals).map(([key, value]) => (
        <div className="signal" key={key}>
          <div style={{ width: "100%" }}>
            <strong>{labels[key as keyof typeof labels]}</strong>
            <div className="bar">
              <span style={{ width: `${Math.round(value * 100)}%` }} />
            </div>
          </div>
          <strong>{Math.round(value * 100)}%</strong>
        </div>
      ))}

      <p className="subtle">
        Triggered: {result.reasons.length ? result.reasons.join(", ") : "No suspicious sub-signal"}
      </p>
    </section>
  );
}