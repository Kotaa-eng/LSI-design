"""Fixed preprocessing and stratified splitting of the bundled Digits data."""

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


INPUT_SCALE = 16.0
CLASS_LABELS = [0, 1]


def load_digits_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Load the full dataset; indices always refer to sklearn's original order."""
    digits = load_digits()
    inputs = np.asarray(digits.images, dtype=np.float32)[:, None, :, :]
    inputs = np.ascontiguousarray(inputs / INPUT_SCALE)
    targets = np.asarray(digits.target, dtype=np.int64)
    return torch.from_numpy(inputs), torch.from_numpy(targets)


def split_binary_indices(
    targets: torch.Tensor, seed: int = 42, test_size: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Select only 0/1 and return disjoint, stratified original dataset indices."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1 (exclusive)")
    target_array = targets.detach().cpu().numpy()
    binary_indices = np.flatnonzero(np.isin(target_array, CLASS_LABELS))
    train_indices, test_indices = train_test_split(
        binary_indices, test_size=test_size, random_state=seed,
        stratify=target_array[binary_indices], shuffle=True,
    )
    return train_indices, test_indices
