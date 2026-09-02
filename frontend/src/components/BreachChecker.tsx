"use client";

import { useState } from "react";

export default function BreachChecker() {
  const [email, setEmail] = useState("");
  const [result, setResult] = useState<string>("");
  const [loading, setLoading] = useState(false);

  async function checkBreach() {
    if (!email) return;
    setLoading(true);
    setResult("");

    try {
      const response = await fetch(`/api/breach-check?email=${encodeURIComponent(email)}`);
      const data = await response.json();

      if (!response.ok) throw new Error(data.error || "Breach lookup failed");

      setResult(data.breached ? `Found in ${data.count} breach record(s). High targeting risk.` : "No breach found.");
    } catch (error) {
      setResult(error instanceof Error ? error.message : "Unable to check breach status.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="panel">
      <h2>Self-Serve Breach Checker</h2>
      <p className="subtle">Shows why a caller or employee may be a fraud target.</p>

      <div className="form">
        <input value={email} onChange={(e) => setEmail(e.target.value)} placeholder="name@company.com" />
        <button className="btn btn-primary" onClick={checkBreach} disabled={loading}>
          {loading ? "Checking..." : "Check"}
        </button>
      </div>

      {result && <div className="result">{result}</div>}
    </section>
  );
}