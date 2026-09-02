import { Mic, Square } from "lucide-react";

type Props = {
  active: boolean;
  loading: boolean;
  onStart: () => void;
  onStop: () => void;
};

export default function CallControls({ active, loading, onStart, onStop }: Props) {
  return (
    <section className="panel">
      <h2>Live Call Protection</h2>
      <p className="subtle">Audio is processed locally in your browser. Raw voice is never uploaded.</p>

      <div className="controls">
        <button className="btn btn-primary" onClick={onStart} disabled={active || loading}>
          <Mic size={18} /> {loading ? "Loading Models..." : "Start Monitoring"}
        </button>

        <button className="btn btn-danger" onClick={onStop} disabled={!active}>
          <Square size={18} /> Stop
        </button>
      </div>
    </section>
  );
}