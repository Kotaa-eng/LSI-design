"""Train the CPU float32 reference CNN on Digits 0/1 with a fixed test split."""

import argparse
from pathlib import Path
import random

import numpy as np
import sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__:
    from .data import CLASS_LABELS, INPUT_SCALE, load_digits_data, split_binary_indices
    from .inference import save_evaluation
    from .model import BinaryDigitsCNN
else:
    from data import CLASS_LABELS, INPUT_SCALE, load_digits_data, split_binary_indices
    from inference import save_evaluation
    from model import BinaryDigitsCNN


PROJECT_DIR = Path(__file__).resolve().parents[1]


def train_model(
    *, output_dir: Path, epochs: int = 40, batch_size: int = 32,
    learning_rate: float = 0.01, seed: int = 42, test_size: float = 0.2,
) -> dict:
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive integers")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    inputs, targets = load_digits_data()
    train_indices, test_indices = split_binary_indices(targets, seed, test_size)
    train_loader = DataLoader(
        TensorDataset(inputs[train_indices], targets[train_indices]),
        batch_size=batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    model = BinaryDigitsCNN()  # CPU float32 is the hardware comparison reference.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_inputs, batch_targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_inputs), batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_targets)
        train_loss = total_loss / len(train_indices)
        history.append({"epoch": epoch, "train_loss": train_loss})
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:3d}/{epochs}: train_loss={train_loss:.6f}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_state_dict": model.state_dict(),
            "seed": seed,
            "test_size": test_size,
            "train_indices": torch.tensor(train_indices, dtype=torch.int64),
            "test_indices": torch.tensor(test_indices, dtype=torch.int64),
            "input_scale": INPUT_SCALE,
            "class_labels": CLASS_LABELS,
        },
        output_dir / "model.pt",
    )
    np.savez_compressed(
        output_dir / "weights.npz",
        conv_weight=model.conv.weight.detach().numpy(),
        conv_bias=model.conv.bias.detach().numpy(),
        fc_weight=model.fc.weight.detach().numpy(),
        fc_bias=model.fc.bias.detach().numpy(),
    )
    metrics = save_evaluation(
        model, inputs[test_indices], targets[test_indices], test_indices, output_dir,
        metadata={
            "seed": seed, "epochs": epochs, "batch_size": batch_size,
            "learning_rate": learning_rate, "test_size": test_size,
            "train_samples": len(train_indices), "test_samples": len(test_indices),
            "class_labels": CLASS_LABELS, "input_scale": INPUT_SCALE,
            "history": history,
            "versions": {
                "torch": str(torch.__version__), "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
            },
        },
    )
    print(f"Test accuracy: {metrics['accuracy']:.4%} ({metrics['correct']}/{metrics['total']})")
    print(f"Training artifacts saved to: {output_dir.resolve()}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "artifacts")
    args = parser.parse_args()
    try:
        train_model(**vars(args))
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
