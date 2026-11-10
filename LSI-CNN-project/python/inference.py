"""Reload a trained CNN and export held-out predictions and every layer output."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

if __package__:
    from .data import CLASS_LABELS, INPUT_SCALE, load_digits_data
    from .model import BinaryDigitsCNN
else:
    from data import CLASS_LABELS, INPUT_SCALE, load_digits_data
    from model import BinaryDigitsCNN


PROJECT_DIR = Path(__file__).resolve().parents[1]


def load_checkpoint(checkpoint_path: Path) -> tuple[BinaryDigitsCNN, dict]:
    """Load tensor/primitive-only checkpoint data on the CPU."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format_version")
    if checkpoint.get("input_scale") != INPUT_SCALE:
        raise ValueError("Checkpoint preprocessing does not match pixel / 16")
    if checkpoint.get("class_labels") != CLASS_LABELS:
        raise ValueError("Checkpoint class labels must be [0, 1]")
    model = BinaryDigitsCNN()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def save_evaluation(
    model: BinaryDigitsCNN,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    dataset_indices: torch.Tensor | np.ndarray,
    output_dir: Path,
    metadata: dict | None = None,
) -> dict:
    """Save all held-out samples in order, including pre-activation outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.inference_mode():
        outputs = model.forward_with_intermediates(inputs)
        predictions = outputs["logits"].argmax(dim=1)
        loss = torch.nn.functional.cross_entropy(outputs["logits"], targets).item()

    arrays = {name: value.detach().cpu().numpy() for name, value in outputs.items()}
    target_array = targets.detach().cpu().numpy()
    prediction_array = predictions.detach().cpu().numpy()
    indices = torch.as_tensor(dataset_indices).cpu().numpy().astype(np.int64)
    if indices.shape != target_array.shape:
        raise ValueError("dataset_indices and targets must have the same shape")
    arrays.update(
        targets=target_array, predictions=prediction_array, dataset_indices=indices,
    )
    np.savez_compressed(output_dir / "intermediates.npz", **arrays)
    correct = int(np.count_nonzero(prediction_array == target_array))
    total = int(target_array.size)
    metrics = dict(metadata or {})
    metrics.update(
        accuracy=correct / total, correct=correct, total=total, loss=loss,
        confusion_matrix=confusion_matrix(
            target_array, prediction_array, labels=CLASS_LABELS,
        ).tolist(),
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=PROJECT_DIR / "artifacts" / "model.pt",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_DIR / "artifacts" / "inference",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model, checkpoint = load_checkpoint(args.checkpoint)
    inputs, targets = load_digits_data()
    indices = checkpoint["test_indices"]
    if (
        not isinstance(indices, torch.Tensor)
        or indices.dtype != torch.int64
        or indices.ndim != 1
        or indices.numel() == 0
        or bool(torch.any((indices < 0) | (indices >= len(targets))))
        or indices.unique().numel() != indices.numel()
    ):
        raise ValueError("Invalid checkpoint test_indices")
    if not bool(torch.all((targets[indices] == 0) | (targets[indices] == 1))):
        raise ValueError("Checkpoint test samples must contain only digits 0 and 1")
    metrics = save_evaluation(
        model, inputs[indices], targets[indices], indices, args.output_dir,
        metadata={"seed": checkpoint["seed"], "checkpoint": str(args.checkpoint)},
    )
    print(f"Test accuracy: {metrics['accuracy']:.4%} ({metrics['correct']}/{metrics['total']})")
    print(f"Inference outputs saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
