
"""Tests for ResNet1D18 using synthetic ECG data.


Contributors:
    Jaspal Bainiwal - netid: jaspalb2
    Anthony Wu      - netid: adwu3
    Pengshao Ye     - netid: py12

Paper: In-depth Benchmarking of Deep Neural Network Architectures for ECG Diagnosis
https://github.com/seitalab/dnn_ecg_comparison


"""

import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.resnet1d_18 import ResNet1D18


LABELS = ["A", "B", "C", "D", "E", "F"]

# Smaller ResNet configuration for fast unit tests.
TINY_MODEL = {
    "base_width": 8,
    "layers": (1, 1, 1, 1),
}


def make_dataset(num_samples: int = 12, length: int = 256):
    """Create a small synthetic multi-label ECG dataset.

    Each sample contains:
        signal: 12-lead ECG with shape (12, length)
        labels: A list of diagnosis labels
    """
    torch.manual_seed(0)

    samples = []

    for i in range(num_samples):
        labels = [LABELS[i % len(LABELS)]]

        # Add another diagnosis to some samples.
        if i % 2 == 0:
            labels.append(LABELS[(i + 2) % len(LABELS)])

        samples.append(
            {
                "patient_id": f"patient_{i}",
                "signal": torch.randn(12, length).tolist(),
                "labels": labels,
            }
        )

    return create_sample_dataset(
        samples=samples,
        input_schema={
            "signal": "tensor",
        },
        output_schema={
            "labels": ("multilabel", {"labels": LABELS}),
        },
    )


class TestResNet1D18(unittest.TestCase):
    """Test model initialization, inference, loss, and gradients."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = make_dataset()

        loader = get_dataloader(
            cls.dataset,
            batch_size=4,
            shuffle=False,
        )

        cls.batch = next(iter(loader))

    def make_model(self):
        """Create a small ResNet1D18 for testing."""
        return ResNet1D18(
            dataset=self.dataset,
            in_channels=12,
            backbone_out_dim=256,
            hidden_dim=128,
            **TINY_MODEL,
        )

    def test_initialization(self):
        """The model should initialize with the dataset's label count."""
        model = self.make_model()

        self.assertEqual(model.in_channels, 12)
        self.assertEqual(model.label_processor.output_size(), len(LABELS))
        self.assertEqual(model.mode, "multilabel")

    def test_forward_shapes_and_probabilities(self):
        """Forward should return correctly shaped logits and probabilities."""
        model = self.make_model()
        model.eval()

        with torch.no_grad():
            output = model(**self.batch)

        expected_shape = (4, len(LABELS))

        self.assertEqual(output["logit"].shape, expected_shape)
        self.assertEqual(output["y_prob"].shape, expected_shape)

        self.assertTrue(torch.isfinite(output["logit"]).all())
        self.assertTrue(torch.isfinite(output["y_prob"]).all())

        self.assertTrue(
            torch.all(output["y_prob"] >= 0)
            and torch.all(output["y_prob"] <= 1)
        )

    def test_loss_is_returned_when_labels_are_provided(self):
        """The model should calculate multi-label loss during training."""
        model = self.make_model()
        model.train()

        output = model(**self.batch)

        self.assertIn("loss", output)
        self.assertIn("y_true", output)

        self.assertEqual(
            output["y_true"].shape,
            output["logit"].shape,
        )

        self.assertTrue(torch.isfinite(output["loss"]))

    def test_forward_without_labels(self):
        """Inference should work without ground-truth labels."""
        model = self.make_model()
        model.eval()

        output = model(signal=self.batch["signal"])

        self.assertIn("logit", output)
        self.assertIn("y_prob", output)
        self.assertNotIn("loss", output)
        self.assertNotIn("y_true", output)

    def test_gradients_reach_backbone(self):
        """Backpropagation should update the backbone and classifier."""
        model = self.make_model()
        model.train()

        output = model(**self.batch)
        output["loss"].backward()

        backbone_grad = model.backbone.stem[0].weight.grad
        classifier_grad = model.fc2.weight.grad

        self.assertIsNotNone(backbone_grad)
        self.assertIsNotNone(classifier_grad)

        self.assertGreater(
            backbone_grad.abs().sum().item(),
            0.0,
        )

        self.assertGreater(
            classifier_grad.abs().sum().item(),
            0.0,
        )

    def test_invalid_input_channels(self):
        """The model should reject input with the wrong number of leads."""
        model = self.make_model()
        model.eval()

        invalid_signal = torch.randn(2, 11, 256)

        with self.assertRaises(ValueError):
            model(signal=invalid_signal)


if __name__ == "__main__":
    unittest.main()
