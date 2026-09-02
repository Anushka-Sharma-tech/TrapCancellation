type AudioFrame =
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
  if (frame instanceof Float32Array) return frame;
  return frame.samples ?? frame.data ?? frame.audio ?? new Float32Array();
}

function getSampleRate(frame: AudioFrame): number {
  if (frame instanceof Float32Array) return 16000;
  return frame.sampleRate ?? 16000;
}

function isSilent(samples: Float32Array): boolean {
  if (samples.length === 0) return true;

  let sumSquares = 0;
  let peak = 0;

  for (const sample of samples) {
    const abs = Math.abs(sample);
    sumSquares += sample * sample;
    peak = Math.max(peak, abs);
  }

  const rms = Math.sqrt(sumSquares / samples.length);

  return rms < 0.012 || peak < 0.035;
}

function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  function writeString(offset: number, value: string) {
    for (let i = 0; i < value.length; i += 1) {
      view.setUint8(offset + i, value.charCodeAt(i));
    }
  }

  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;

  for (const sample of samples) {
    const value = Math.max(-1, Math.min(1, sample));
    view.setInt16(offset, value < 0 ? value * 0x8000 : value * 0x7fff, true);
    offset += 2;
  }

  return new Blob([view], { type: "audio/wav" });
}

export async function runVoiceInference(frame: AudioFrame): Promise<VoiceSignals> {
  const samples = getSamples(frame);

  if (isSilent(samples)) {
    return emptySignals;
  }

  const wav = encodeWav(samples, getSampleRate(frame));

  const response = await fetch("/api/voice/analyze", {
    method: "POST",
    headers: {
      "Content-Type": "audio/wav",
    },
    body: wav,
  });

  if (!response.ok) {
    throw new Error("Voice inference failed");
  }

  return response.json();
}