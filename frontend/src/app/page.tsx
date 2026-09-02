"use client";

import { useRef, useState } from "react";
import RiskGauge from "@/components/RiskGauge";
import CallControls from "@/components/CallControls";
import AlertPanel from "@/components/AlertPanel";
import BreachChecker from "@/components/BreachChecker";
import { AudioProcessor } from "@/lib/audioProcessor";
import { runVoiceInference } from "@/lib/inference";
import { calculateRisk, RiskResult } from "@/lib/riskScorer";

const noSpeechResult: RiskResult = {
  score: 0,
  level: "LOW",
  signals: {
    acousticArtifact: 0,
    prosodyAnomaly: 0,
    speakerDrift: 0,
    behavioralRisk: 0,
  },
  reasons: ["Waiting for clear speech. Silence and room noise are ignored."],
};

const capabilities = [
  "UPI and net banking call protection",
  "Voice deepfake and spoof detection",
  "No raw audio stored by default",
  "Explainable risk scoring for audit teams",
];

const bankingUseCases = [
  "Customer care fraud calls",
  "Loan and KYC voice checks",
  "Relationship manager impersonation",
  "Telecom-scale call screening",
];

export default function Home() {
  const processor = useRef<AudioProcessor | null>(null);
  const busy = useRef(false);

  const [active, setActive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [risk, setRisk] = useState<RiskResult>(noSpeechResult);

  async function start() {
    setLoading(true);
    processor.current = new AudioProcessor();

    await processor.current.start(async (frame) => {
      if (busy.current) return;

      busy.current = true;

      try {
        const signals = await runVoiceInference(frame);
        setRisk(calculateRisk(signals));
      } catch {
        setRisk((previous) => previous);
      } finally {
        busy.current = false;
      }
    });

    setActive(true);
    setLoading(false);
  }

  function stop() {
    processor.current?.stop();
    processor.current = null;
    setActive(false);
    setRisk(noSpeechResult);
  }

  const reasons = risk.reasons.length > 0 ? risk.reasons : noSpeechResult.reasons;

  return (
    <main className="frontpage">
      <section className="hero">
        <nav className="nav">
          <div className="logo-mark">TC</div>
          <span className="brand-title">TrapCancellation</span>
          <div className="nav-pill">Bharat BFSI Voice Security</div>
        </nav>

        <div className="hero-grid">
          <div className="hero-copy">
            <p className="eyebrow">
              Smart India Hackathon ready voice integrity framework
            </p>

            <h1 className="hero-title">
              <span>Trap</span>
              <span>Cancellation</span>
            </h1>

            <p className="hero-text">
              Real-time deepfake call defense for Indian banking, fintech,
              telecom, and enterprise environments. Detect cloned, AI-generated,
              or manipulated voices with explainable impersonation risk scoring.
            </p>

            <div className="capability-row">
              {capabilities.map((item) => (
                <span key={item}>{item}</span>
              ))}
            </div>

            <div className="hero-controls panel glass-panel">
              <CallControls
                active={active}
                loading={loading}
                onStart={start}
                onStop={stop}
              />

              <p className="microcopy">
                Mic frames are checked for real speech first. Silence is not
                treated as fraud.
              </p>
            </div>
          </div>

          <aside className="risk-card panel">
            <div className="chakra-shield">
              <div className="shield-ring">
                <span>₹</span>
              </div>
            </div>

            <div className="risk-topline">
              <span>Dynamic Impersonation Risk</span>
              <strong>{risk.score}/100</strong>
            </div>

            <div className={`risk-status risk-${risk.level.toLowerCase()}`}>
              {risk.level} RISK
            </div>

            <div className="reason-box">
              <h2>Why this risk is occurring</h2>

              <ul>
                {reasons.map((reason) => (
                  <li key={reason}>{reason}</li>
                ))}
              </ul>
            </div>
          </aside>
        </div>
      </section>

      <section className="shell">
        <div className="section-heading">
          <p className="eyebrow">Live Protection Console</p>
          <h2>Explainable voice fraud intelligence</h2>
        </div>

        <div className="grid">
          <div className="stack">
            <RiskGauge result={risk} />

            <div className="panel usecase-panel">
              <h3>Built for banks, fintech, telecom, and enterprises</h3>

              <div className="usecase-grid">
                {bankingUseCases.map((item) => (
                  <div className="usecase-item" key={item}>
                    <span>✓</span>
                    {item}
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="stack">
            <AlertPanel result={risk} />

            <div className="panel privacy-panel">
              <h3>Privacy protected by design</h3>

              <p>
                Use browser-side screening for fast local checks and a secured
                backend only for advanced deep learning inference. Raw audio can
                remain ephemeral unless audit retention is explicitly required.
              </p>
            </div>

            <BreachChecker />
          </div>
        </div>
      </section>
    </main>
  );
}