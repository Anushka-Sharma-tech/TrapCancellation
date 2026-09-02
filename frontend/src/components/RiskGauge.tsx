import { RiskResult } from "../lib/riskScorer";

export default function RiskGauge({ result }: { result: RiskResult }) {
  const color =
    result.level === "HIGH" ? "#d92d20" : result.level === "MEDIUM" ? "#dc6803" : "#039855";

  return (
    <section className="panel gauge">
      <div
        className="gauge-ring"
        style={{
          background: `conic-gradient(${color} ${result.score * 3.6}deg, #eaecf0 0deg)`
        }}
      >
        <div className="gauge-inner">
          <div>
            <div className="score">{result.score}</div>
            <div className={`alert-${result.level.toLowerCase()}`}>{result.level} RISK</div>
          </div>
        </div>
      </div>
      <p className="subtle">Explainable fraud risk from on-device audio inference</p>
    </section>
  );
}