export type SignalScores = {
  acousticArtifact: number;
  prosodyAnomaly: number;
  speakerDrift: number;
  behavioralRisk: number;
};

export type RiskResult = {
  score: number;
  level: "LOW" | "MEDIUM" | "HIGH";
  signals: SignalScores;
  reasons: string[];
};

const clamp01 = (value: number) => Math.max(0, Math.min(1, value));

export function calculateRisk(signals: SignalScores): RiskResult {
  const normalized = {
    acousticArtifact: clamp01(signals.acousticArtifact),
    prosodyAnomaly: clamp01(signals.prosodyAnomaly),
    speakerDrift: clamp01(signals.speakerDrift),
    behavioralRisk: clamp01(signals.behavioralRisk)
  };

  const score =
    normalized.acousticArtifact * 0.42 +
    normalized.prosodyAnomaly * 0.25 +
    normalized.speakerDrift * 0.23 +
    normalized.behavioralRisk * 0.1;

  const reasons: string[] = [];

  if (normalized.acousticArtifact > 0.62) reasons.push("Acoustic Artifact");
  if (normalized.prosodyAnomaly > 0.58) reasons.push("Prosody Anomaly");
  if (normalized.speakerDrift > 0.55) reasons.push("Speaker Drift");
  if (normalized.behavioralRisk > 0.6) reasons.push("Behavioral Risk");

  return {
    score: Math.round(score * 100),
    level: score >= 0.7 ? "HIGH" : score >= 0.4 ? "MEDIUM" : "LOW",
    signals: normalized,
    reasons
  };
}