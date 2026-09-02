import torch
from pathlib import Path


class PlaceholderWav2VecDeepfake(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv1d(1, 16, kernel_size=9, stride=2, padding=4),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(128),
            torch.nn.Flatten(),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(32 * 128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2),
        )

    def forward(self, input):
        x = input.unsqueeze(1)
        x = self.feature_extractor(x)
        return self.classifier(x)


def export():
    output = Path(__file__).resolve().parent.parent / "frontend" / "public" / "models" / "wav2vec2-deepfake.onnx"
    output.parent.mkdir(parents=True, exist_ok=True)

    model = PlaceholderWav2VecDeepfake().eval()
    dummy = torch.randn(1, 16000)

    torch.onnx.export(
        model,
        dummy,
        output,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch", 1: "samples"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )

    print(f"Exported: {output}")
    print(f"Size: {output.stat().st_size} bytes")


if __name__ == "__main__":
    export()