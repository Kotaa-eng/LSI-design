"""Numerical and end-to-end checks for the binary digits CNN.

Run from LSI-CNN-project with: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from python.data import load_digits_data, split_binary_indices  # noqa: E402
from python.inference import load_checkpoint, save_evaluation  # noqa: E402
from python.model import BinaryDigitsCNN  # noqa: E402
from python.train import train_model  # noqa: E402


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.model = BinaryDigitsCNN().eval()

    def test_architecture_and_intermediate_shapes(self):
        inputs = torch.zeros(2, 1, 8, 8)
        expected_shapes = {
            "input": (2, 1, 8, 8),
            "conv": (2, 4, 6, 6),
            "relu": (2, 4, 6, 6),
            "pool": (2, 4, 3, 3),
            "flatten": (2, 36),
            "logits": (2, 2),
        }
        with torch.inference_mode():
            outputs = self.model.forward_with_intermediates(inputs)
            logits = self.model(inputs)
        self.assertEqual(set(outputs), set(expected_shapes))
        for name, shape in expected_shapes.items():
            with self.subTest(layer=name):
                self.assertEqual(tuple(outputs[name].shape), shape)
        torch.testing.assert_close(logits, outputs["logits"], rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 114)
        self.assertEqual(self.model.conv.stride, (1, 1))
        self.assertEqual(self.model.conv.padding, (0, 0))
        self.assertEqual(self.model.conv.dilation, (1, 1))
        self.assertEqual(self.model.conv.groups, 1)
        self.assertIsNotNone(self.model.conv.bias)
        self.assertIsNotNone(self.model.fc.bias)

    def test_convolution_is_cross_correlation_with_bias(self):
        # A non-symmetric kernel distinguishes correlation from kernel flipping.
        with torch.no_grad():
            self.model.conv.weight.zero_()
            self.model.conv.bias.zero_()
            self.model.conv.weight[0, 0] = torch.arange(1, 10).reshape(3, 3)
            self.model.conv.bias[0] = -1
        inputs = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)
        with torch.inference_mode():
            conv = self.model.forward_with_intermediates(inputs)["conv"]
        # 0*1 + 1*2 + 2*3 + 8*4 + 9*5 + 10*6
        # + 16*7 + 17*8 + 18*9 - 1 = 554.
        self.assertEqual(conv[0, 0, 0, 0].item(), 554.0)
        # Moving one input column adds the sum of the kernel coefficients.
        self.assertEqual(conv[0, 0, 0, 1].item(), 599.0)
        self.assertEqual(torch.count_nonzero(conv[:, 1:]).item(), 0)

    def test_relu_pool_windows_and_channel_major_flatten(self):
        with torch.inference_mode():
            relu = self.model.relu(torch.tensor([-2.0, 0.0, 3.0]))
            activation = torch.arange(144, dtype=torch.float32).reshape(1, 4, 6, 6)
            pooled = self.model.pool(activation)
            flattened = self.model.flatten(pooled)
        torch.testing.assert_close(relu, torch.tensor([0.0, 0.0, 3.0]))
        expected = np.array(
            [36 * channel + 6 * (2 * row + 1) + 2 * col + 1
             for channel in range(4) for row in range(3) for col in range(3)],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(pooled.numpy(), expected.reshape(1, 4, 3, 3))
        np.testing.assert_array_equal(flattened.numpy(), expected.reshape(1, 36))

    def test_linear_equation_and_class_zero_tie_break(self):
        with torch.no_grad():
            self.model.fc.weight[0] = torch.arange(36)
            self.model.fc.weight[1].fill_(-1)
            self.model.fc.bias.copy_(torch.tensor([1.5, -2.5]))
        with torch.inference_mode():
            logits = self.model.fc(torch.arange(36, dtype=torch.float32).reshape(1, 36))
        torch.testing.assert_close(logits, torch.tensor([[14911.5, -632.5]]), rtol=0, atol=0)
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.zero_()
        with torch.inference_mode():
            tied_logits = self.model(torch.ones(2, 1, 8, 8))
        torch.testing.assert_close(tied_logits, torch.zeros(2, 2), rtol=0, atol=0)
        self.assertEqual(tied_logits.argmax(dim=1).tolist(), [0, 0])


class DataTests(unittest.TestCase):
    def test_normalization_and_reproducible_disjoint_binary_split(self):
        inputs, targets = load_digits_data()
        self.assertEqual(tuple(inputs.shape), (len(targets), 1, 8, 8))
        self.assertEqual(inputs.dtype, torch.float32)
        self.assertEqual(targets.dtype, torch.int64)
        self.assertEqual(inputs.min().item(), 0.0)
        self.assertEqual(inputs.max().item(), 1.0)
        torch.testing.assert_close(inputs * 16, (inputs * 16).round(), rtol=0, atol=0)
        train_indices, test_indices = split_binary_indices(targets, seed=42, test_size=0.2)
        train_again, test_again = split_binary_indices(targets, seed=42, test_size=0.2)
        np.testing.assert_array_equal(train_indices, train_again)
        np.testing.assert_array_equal(test_indices, test_again)
        self.assertEqual(train_indices.dtype, np.int64)
        self.assertEqual(test_indices.dtype, np.int64)
        self.assertEqual(len(np.intersect1d(train_indices, test_indices)), 0)
        binary_indices = np.flatnonzero(np.isin(targets.numpy(), [0, 1]))
        np.testing.assert_array_equal(
            np.sort(np.concatenate([train_indices, test_indices])), binary_indices
        )
        self.assertEqual(set(targets[train_indices].tolist()), {0, 1})
        self.assertEqual(set(targets[test_indices].tolist()), {0, 1})
        _, different_test = split_binary_indices(targets, seed=43, test_size=0.2)
        self.assertFalse(np.array_equal(test_indices, different_test))


class TrainingIntegrationTests(unittest.TestCase):
    def test_training_checkpoint_and_saved_evaluation_agree(self):
        with TemporaryDirectory(prefix="binary-digits-test-") as temporary:
            output_dir = Path(temporary) / "training"
            train_model(output_dir=output_dir, epochs=2, batch_size=32, seed=17)
            self.assertTrue((output_dir / "model.pt").is_file())
            self.assertTrue((output_dir / "metrics.json").is_file())
            model, checkpoint = load_checkpoint(output_dir / "model.pt")
            self.assertEqual(checkpoint["seed"], 17)
            self.assertEqual(checkpoint["format_version"], 1)
            self.assertEqual(checkpoint["input_scale"], 16.0)
            self.assertEqual(checkpoint["class_labels"], [0, 1])
            self.assertIn("model_state_dict", checkpoint)
            inputs, targets = load_digits_data()
            train_indices = checkpoint["train_indices"]
            test_indices = checkpoint["test_indices"]
            self.assertEqual(len(np.intersect1d(train_indices, test_indices)), 0)
            with torch.inference_mode():
                reference = model.forward_with_intermediates(inputs[test_indices])
            evaluation_dir = Path(temporary) / "evaluation"
            metrics = save_evaluation(
                model,
                inputs[test_indices],
                targets[test_indices],
                test_indices,
                evaluation_dir,
                metadata={"test": "checkpoint_round_trip"},
            )
            predictions = reference["logits"].argmax(dim=1)
            expected_correct = int((predictions == targets[test_indices]).sum().item())
            self.assertEqual(metrics["total"], len(test_indices))
            self.assertEqual(metrics["correct"], expected_correct)
            self.assertAlmostEqual(metrics["accuracy"], expected_correct / len(test_indices))
            expected_confusion = np.zeros((2, 2), dtype=np.int64)
            np.add.at(expected_confusion, (targets[test_indices].numpy(), predictions.numpy()), 1)
            np.testing.assert_array_equal(metrics["confusion_matrix"], expected_confusion)
            with (evaluation_dir / "metrics.json").open(encoding="utf-8") as file:
                saved_metrics = json.load(file)
            self.assertEqual(saved_metrics["correct"], expected_correct)
            self.assertEqual(saved_metrics["total"], len(test_indices))
            with np.load(evaluation_dir / "intermediates.npz", allow_pickle=False) as saved:
                self.assertEqual(
                    set(saved.files),
                    set(reference) | {"targets", "predictions", "dataset_indices"},
                )
                for name, expected_tensor in reference.items():
                    with self.subTest(layer=name):
                        np.testing.assert_allclose(saved[name], expected_tensor.numpy(), rtol=1e-6, atol=1e-6)
                np.testing.assert_array_equal(saved["targets"], targets[test_indices].numpy())
                np.testing.assert_array_equal(saved["predictions"], predictions.numpy())
                np.testing.assert_array_equal(saved["dataset_indices"], test_indices.numpy())
            with np.load(output_dir / "weights.npz", allow_pickle=False) as weights:
                self.assertEqual(set(weights.files), {"conv_weight", "conv_bias", "fc_weight", "fc_bias"})
                for name, parameter in {
                    "conv_weight": model.conv.weight,
                    "conv_bias": model.conv.bias,
                    "fc_weight": model.fc.weight,
                    "fc_bias": model.fc.bias,
                }.items():
                    np.testing.assert_array_equal(weights[name], parameter.detach().numpy())


if __name__ == "__main__":
    unittest.main()
