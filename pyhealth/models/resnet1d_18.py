"""
ResNet1D-18 Model Implementation

Contributors:
    Jaspal Bainiwal - netid: jaspalb2
    Anthony Wu      - netid: adwu3
    Pengshao Ye     - netid: py12

Paper: In-depth Benchmarking of Deep Neural Network Architectures for ECG Diagnosis
https://github.com/seitalab/dnn_ecg_comparison
"""


"""
ResNet-18 for multi-label ECG classification.

Adapted from:
"In-Depth Benchmarking of Deep Neural Network Architectures for ECG Diagnosis"

The model expects ECG windows with shape:
    (batch_size, 12, sequence_length)

For the Georgia ECG experiment:
    sequence_length = 1250  # 2.5 seconds at 500 Hz
    number of labels = 30
"""

import torch
from torch import nn

from pyhealth.datasets import SampleDataset

from .base_model import BaseModel
from .protoecgnet import ResNet1dBackbone


class ResNet1D18(BaseModel):
    """1D ResNet-18 for multi-label ECG classification.

    The architecture consists of:
        1. A 1D ResNet-18 backbone.
        2. Global average pooling.
        3. A projection layer (512 -> 256).
        4. A classification head (256 -> 128 -> num_labels).

    Args:
        dataset: PyHealth SampleDataset used to configure input and output
            schemas and retrieve the label processor.
        in_channels: Number of ECG leads. Defaults to 12.
        backbone_out_dim: Dimension of the projected backbone features.
            Defaults to 256, matching the reference implementation.
        hidden_dim: Hidden dimension of the classification head.
            Defaults to 128.
        dropout: Dropout probability in the classification head.
            Defaults to 0.25.
        feature_key: Name of the ECG input field in the dataset.
            Defaults to "signal".
        label_key: Name of the target label field in the dataset.
            Defaults to "labels".

    Examples:
        >>> model = ResNet1D18(dataset=sample_dataset)
        >>> output = model(signal=torch.randn(4, 12, 1250))
        >>> output["logit"].shape
        torch.Size([4, 30])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        in_channels: int = 12,
        backbone_out_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.25,
        feature_key: str = "signal",
        label_key: str = "labels",
        base_width: int = 64,
        layers: tuple[int, int, int, int] = (2, 2, 2, 2),
    ):
        super().__init__(dataset=dataset)

        # --------------------------------------------------
        # Validate the dataset configuration
        # --------------------------------------------------

        if feature_key not in self.feature_keys:
            raise ValueError(
                f"Input feature '{feature_key}' not found in dataset. "
                f"Available features: {self.feature_keys}"
            )

        if label_key not in self.label_keys:
            raise ValueError(
                f"Output label '{label_key}' not found in dataset. "
                f"Available labels: {self.label_keys}"
            )

        if self.label_processor.mode != "multilabel":
            raise ValueError(
                "ResNet1D18 expects a multi-label classification task, "
                f"but received mode '{self.label_processor.mode}'."
            )

        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")

        if backbone_out_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Feature dimensions must be positive.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        self.in_channels = in_channels
        self.feature_key = feature_key
        self.label_key = label_key

        num_labels = self.label_processor.output_size()

        # --------------------------------------------------
        # 1. ResNet-18 backbone
        # --------------------------------------------------
        #
        # The existing PyHealth backbone implements:
        #   - 7x1 convolutional stem
        #   - Max pooling
        #   - Four residual stages
        #   - [2, 2, 2, 2] BasicBlock configuration
        #
        # Its output has shape (B, 512, T').

        self.backbone = ResNet1dBackbone(
            in_channels=in_channels,
            base_width=base_width,
            layers=layers,
            kernel_size=3,
        )

        # Initialize the backbone using the initialization
        # approach from the reference ResNet implementation.
        self._initialize_backbone()

        # --------------------------------------------------
        # 2. Global average pooling
        # --------------------------------------------------

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # --------------------------------------------------
        # 3. Backbone projection
        # --------------------------------------------------
        #
        # (B, 512) -> (B, 256)

        self.backbone_fc = nn.Linear(
            self.backbone.out_channels,
            backbone_out_dim,
        )

        # --------------------------------------------------
        # 4. Classification head
        # --------------------------------------------------
        #
        # (B, 256) -> (B, 128) -> (B, num_labels)

        self.fc1 = nn.Linear(backbone_out_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, num_labels)

    def _initialize_backbone(self) -> None:
        """Initialize convolutional and normalization layers.

        Uses Kaiming initialization for convolutional layers and
        initializes batch normalization weights to 1 and biases to 0.
        """

        for module in self.backbone.modules():

            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Run a forward pass.

        Args:
            **kwargs: Input tensors supplied by the PyHealth data loader.
                Expected keys include:
                    signal: ECG tensor with shape (B, 12, T).
                    labels: Optional multi-label target tensor (B, num_labels).

        Returns:
            A dictionary containing:
                logit: Raw classification logits.
                y_prob: Sigmoid probabilities.
                loss: Multi-label loss, when labels are supplied.
                y_true: Ground-truth labels, when labels are supplied.
        """

        # --------------------------------------------------
        # Retrieve and validate ECG input
        # --------------------------------------------------

        if self.feature_key not in kwargs:
            raise KeyError(
                f"Missing input '{self.feature_key}' in forward()."
            )

        x = kwargs[self.feature_key]

        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"Expected '{self.feature_key}' to be a torch.Tensor, "
                f"but received {type(x).__name__}."
            )

        if x.ndim != 3:
            raise ValueError(
                "Expected ECG input with shape (B, C, T), "
                f"but received shape {tuple(x.shape)}."
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} ECG leads, "
                f"but received {x.shape[1]}."
            )

        x = x.to(device=self.device, dtype=torch.float32)

        # --------------------------------------------------
        # ResNet-18 feature extraction
        # --------------------------------------------------

        x = self.backbone(x)
        # Shape: (B, 512, T')

        # --------------------------------------------------
        # Global average pooling
        # --------------------------------------------------

        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        # Shape: (B, 512)

        # --------------------------------------------------
        # Project to backbone_out_dim
        # --------------------------------------------------

        x = self.backbone_fc(x)
        # Shape: (B, 256)

        # --------------------------------------------------
        # Classification head
        # --------------------------------------------------

        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.dropout(x)

        logits = self.fc2(x)
        # Shape: (B, num_labels)

        # --------------------------------------------------
        # Convert logits into probabilities
        # --------------------------------------------------
        #
        # The MultiLabelProcessor applies sigmoid activation.
        # Do not apply sigmoid manually before calling it.

        y_prob = self.label_processor.activation(logits)

        output = {
            "logit": logits,
            "y_prob": y_prob,
        }

        # --------------------------------------------------
        # Calculate loss when ground-truth labels exist
        # --------------------------------------------------

        if self.label_key in kwargs and kwargs[self.label_key] is not None:

            y_true = kwargs[self.label_key]

            if not isinstance(y_true, torch.Tensor):
                raise TypeError(
                    f"Expected '{self.label_key}' to be a torch.Tensor."
                )

            y_true = y_true.to(
                device=self.device,
                dtype=logits.dtype,
            )

            if y_true.shape != logits.shape:
                raise ValueError(
                    "Ground-truth labels and logits must have the same shape. "
                    f"Received labels {tuple(y_true.shape)} and "
                    f"logits {tuple(logits.shape)}."
                )

            loss = self.label_processor.loss(logits, y_true)

            output["loss"] = loss
            output["y_true"] = y_true

        return output
