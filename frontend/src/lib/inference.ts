export type AudioFrame =
  | Float32Array
  | {
      samples?: Float32Array;
      data?: Float32Array;
      audio?: Float32Array;
      sampleRate?: number;
    };

export type VoiceSignals = {
  acousticArtifact: number;
  prosodyAnomaly: number;
  speakerDrift: number;
  behavioralRisk: number;
};

const emptySignals: VoiceSignals = {
  acousticArtifact: 0,
  prosodyAnomaly: 0,
  speakerDrift: 0,
  behavioralRisk: 0,
};

function getSamples(frame: AudioFrame): Float32Array {
  if (frame instanceof Float32Array) {
    return frame;
  }

  return (
    frame.samples ??
    frame.data ??
    frame.audio ??
    new Float32Array()
  );
}

function isSilent(samples: Float32Array): boolean {
  if (samples.length === 0) {
    return true;
  }

  let sumSquares = 0;
  let peak = 0;

  for (const sample of samples) {
    const value = Math.abs(sample);

    sumSquares += sample * sample;
    peak = Math.max(peak, value);
  }

  const rms = Math.sqrt(sumSquares / samples.length);

  return rms < 0.012 || peak < 0.035;
}

function getBackendUrl(): string {
  const url = process.env.NEXT_PUBLIC_VOICE_API_URL?.trim();

  if (!url) {
    throw new Error(
      "NEXT_PUBLIC_VOICE_API_URL is not configured."
    );
  }

  return url.replace(/\/+$/, "");
}

export async function runVoiceInference(
  frame: AudioFrame
): Promise<VoiceSignals> {
  const samples = getSamples(frame);

  if (isSilent(samples)) {
    return emptySignals;
  }

  const backendUrl = getBackendUrl();

  /*
   * /stream/analyze expects:
   * - raw Float32 PCM bytes
   * - 16 kHz
   *
   * slice() creates an exact-sized copy so we don't accidentally send
   * unrelated bytes from the underlying ArrayBuffer.
   */
  const pcmBuffer = samples.slice().buffer;

  const response = await fetch(
    `${backendUrl}/stream/analyze`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/octet-stream",
      },
      body: pcmBuffer,
      cache: "no-store",
    }
  );

  if (!response.ok) {
    const errorText = await response.text().catch(() => "");

    throw new Error(
      `Voice inference failed (${response.status})${
        errorText ? `: ${errorText}` : ""
      }`
    );
  }

  const result = await response.json();

  return {
    /*
     * The backend streaming response uses 0-100 scores.
     * The existing frontend signal model uses 0-1 values.
     */
    acousticArtifact:
      Number(result.acousticScore ?? 0) / 100,

    prosodyAnomaly:
      Number(result.prosodyScore ?? 0) / 100,

    speakerDrift:
      Number(result.speakerConsistencyScore ?? 0) / 100,

    behavioralRisk: 0,
  };
}