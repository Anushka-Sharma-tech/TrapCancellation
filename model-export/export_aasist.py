import torch
from pathlib import Path


class PlaceholderAASIST(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(1, 32, kernel_size=9, stride=2, padding=4),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(64, 2),
        )

    def forward(self, input):
        x = input.unsqueeze(1)
        return self.net(x)


def export():
    output = Path("../frontend/public/models/aasist.onnx")
    output.parent.mkdir(parents=True, exist_ok=True)

    model = PlaceholderAASIST().eval()
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

    print(f"Exported {output}")


if __name__ == "__main__":
    export()