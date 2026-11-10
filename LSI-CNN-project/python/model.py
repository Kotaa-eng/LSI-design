"""Small 0/1 digit classifier with explicitly observable layer boundaries."""

import torch
from torch import nn


class BinaryDigitsCNN(nn.Module):
    """NCHW: (N, 1, 8, 8) -> (N, 4, 6, 6) -> (N, 4, 3, 3) -> (N, 2)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=1, out_channels=4, kernel_size=3,
            stride=1, padding=0, dilation=1, groups=1, bias=True,
        )
        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(in_features=36, out_features=2, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(self.flatten(self.pool(self.relu(self.conv(inputs)))))

    def forward_with_intermediates(
        self, inputs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return each layer's output, preserving the pre-ReLU convolution."""
        outputs = {"input": inputs}
        outputs["conv"] = self.conv(outputs["input"])
        outputs["relu"] = self.relu(outputs["conv"])
        outputs["pool"] = self.pool(outputs["relu"])
        outputs["flatten"] = self.flatten(outputs["pool"])
        outputs["logits"] = self.fc(outputs["flatten"])
        return outputs
