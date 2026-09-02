export type AudioFrameHandler = (frame: Float32Array) => void;

export class AudioProcessor {
  private context?: AudioContext;
  private source?: MediaStreamAudioSourceNode;
  private processor?: ScriptProcessorNode;
  private stream?: MediaStream;

  async start(onFrame: AudioFrameHandler) {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        noiseSuppression: true,
        echoCancellation: true,
        autoGainControl: false
      }
    });

    this.context = new AudioContext({ sampleRate: 16000 });
    this.source = this.context.createMediaStreamSource(this.stream);
    this.processor = this.context.createScriptProcessor(4096, 1, 1);

    this.processor.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0);
      onFrame(new Float32Array(input));
    };

    this.source.connect(this.processor);
    this.processor.connect(this.context.destination);
  }

  stop() {
    this.processor?.disconnect();
    this.source?.disconnect();
    this.stream?.getTracks().forEach((track) => track.stop());
    void this.context?.close();

    this.context = undefined;
    this.source = undefined;
    this.processor = undefined;
    this.stream = undefined;
  }
}

export function extractProsodyFeatures(frame: Float32Array) {
  let energy = 0;
  let zeroCrossings = 0;

  for (let i = 0; i < frame.length; i++) {
    energy += frame[i] * frame[i];
    if (i > 0 && Math.sign(frame[i]) !== Math.sign(frame[i - 1])) zeroCrossings++;
  }

  const rms = Math.sqrt(energy / frame.length);
  const zcr = zeroCrossings / frame.length;

  return {
    rms,
    zcr,
    prosodyAnomaly: Math.min(1, Math.abs(rms - 0.04) * 8 + Math.abs(zcr - 0.08) * 3)
  };
}