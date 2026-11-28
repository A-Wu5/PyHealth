"""Georgia ECG multilabel classification with ResNet1D18.

Reproduces the G12EC ResNet-18 setting of Nonaka and Seita (2021) using
PyHealth's Georgia data loader and ResNet1D18. The paper reports a Macro
ROC-AUC of 0.979 +/- 0.002 over five independent runs. This example uses
the saved seed-1 split; its formal result and aggregation ablation are Pending.
Paper: https://proceedings.mlr.press/v149/nonaka21a.html

Real-data setup: prepare ``data/georgia/g1`` through ``g11`` and the three
files in ``data/georgia_processed`` using the team's data-processing workflow.
Use corrected training-set normalization statistics before a real run.
No raw ECG download or preprocessing is performed here.

    python examples/georgia_ecg_classification_resnet1d18.py --synthetic
    python examples/georgia_ecg_classification_resnet1d18.py \
        --data-root data --processed-root data/georgia_processed

Real configuration: seed 1; Adam, learning rate 0.001; batch size 64;
up to 250 epochs; full Validation every 5 epochs; patience 5 checks.
The best model is selected by Validation Max Macro ROC-AUC and then loaded
for one Test inference. Max is the paper baseline; Mean and Top-2 Mean are
our aggregation ablation. All three use the same Test window probabilities.
Synthetic scores only check that the pipeline works; they are not ECG results.
Observed local CPU demo (seed 1, two epochs): best Validation epoch 1;
Test Macro ROC-AUC: Max 0.4398, Mean 0.3981, Top-2 Mean 0.4398.
These values are not the Pending G12EC reproduction results.
"""

import argparse
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from pyhealth.datasets import create_sample_dataset
from pyhealth.models.resnet1d_18 import ResNet1D18
from pyhealth.tasks.georgia_ecg import load_georgia_data


def predict_dataset(model, loader, codes, device):
    all_probabilities, all_labels, all_ids = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_size, windows = batch["signal"].shape[:2]
            signal = batch["signal"].reshape(batch_size * windows, 12, 1250).to(device)
            y_prob = model(signal=signal)["y_prob"]
            all_probabilities.append(
                y_prob.reshape(batch_size, windows, len(codes)).cpu()
            )
            all_labels.append(batch["labels"].cpu())
            all_ids.extend(batch["record_id"])
    probabilities = torch.cat(all_probabilities)
    labels = torch.cat(all_labels).numpy()
    assert len(all_ids) == len(set(all_ids)) == len(loader.dataset)
    assert all_ids == [
        loader.dataset[i]["record_id"] for i in range(len(loader.dataset))
    ]
    assert probabilities.shape == (len(all_ids), 8, len(codes))
    assert labels.shape == (len(all_ids), len(codes))
    assert np.isin(labels, [0, 1]).all()
    assert torch.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    return probabilities, labels, np.asarray(all_ids)


def aggregate_predictions(probabilities):
    return {
        "Max": probabilities.max(dim=1).values,
        "Mean": probabilities.mean(dim=1),
        "Top-2 Mean": probabilities.topk(2, dim=1).values.mean(dim=1),
    }


def evaluate_auc(y_true, probabilities, codes):
    scores = np.asarray(probabilities)
    assert scores.shape == y_true.shape == (len(y_true), len(codes))
    assert np.isin(y_true, [0, 1]).all() and np.isfinite(scores).all()
    per_class = pd.Series(
        [roc_auc_score(y_true[:, i], scores[:, i]) for i in range(len(codes))],
        index=codes, name="ROC-AUC"
    )
    assert len(per_class) == len(codes) and np.isfinite(per_class).all()
    return per_class.mean(), per_class


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--synthetic", action="store_true", help="quick CPU demo")
    parser.add_argument("--data-root", help="prepared data directory")
    parser.add_argument("--processed-root", help="saved split and normalization files")
    parser.add_argument("--epochs", type=int, help="default: 2 synthetic, 250 real")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if not args.synthetic and (not args.data_root or not args.processed_root):
        parser.error("pass --synthetic or both --data-root and --processed-root")

    toy = torch.tensor([[[0.1, 0.9], [0.8, 0.2], [0.6, 0.7]]])
    toy_predictions = aggregate_predictions(toy)
    assert torch.allclose(toy_predictions["Max"], torch.tensor([[0.8, 0.9]]))
    assert torch.allclose(toy_predictions["Mean"], torch.tensor([[0.5, 0.6]]))
    assert torch.allclose(toy_predictions["Top-2 Mean"], torch.tensor([[0.7, 0.8]]))
    toy_labels = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
    toy_scores = np.array([[0.1, 0.9], [0.9, 0.1], [0.2, 0.8], [0.8, 0.2]])
    assert evaluate_auc(toy_labels, toy_scores, ["a", "b"])[0] == 1.0

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cpu" if args.synthetic else "cuda")
    if not args.synthetic and not torch.cuda.is_available():
        parser.error("real training requires a CUDA GPU")

    if args.synthetic:
        codes = ["A", "B", "C"]
        raw_samples = []
        for i in range(24):
            labels = [codes[i % len(codes)]]
            if i % 4 == 0:
                labels.append(codes[(i + 1) % len(codes)])
            raw_samples.append({
                "patient_id": f"patient_{i}",
                "record_id": f"demo_{i}",
                "signal": torch.randn(12, 1250).tolist(),
                "labels": labels,
            })
        samples = create_sample_dataset(
            samples=raw_samples,
            input_schema={"signal": "tensor"},
            output_schema={"labels": ("multilabel", {"labels": codes})},
        )
        train = [samples[i] for i in range(12)]
        val, test = [], []
        for i in range(12, 24):
            sample = dict(samples[i])
            signal = sample["signal"]
            sample["signal"] = torch.stack([signal + 0.01 * j for j in range(8)])
            (val if i < 18 else test).append(sample)
        for dataset in (val, test):
            counts = torch.stack([sample["labels"] for sample in dataset]).sum(dim=0)
            assert ((counts > 0) & (counts < len(dataset))).all()
        model_options = {"base_width": 8, "layers": (1, 1, 1, 1)}
        epochs, batch_size, eval_batch_size = args.epochs or 2, 4, 2
        validate_every = 1
    else:
        train, val, test, codes, samples = load_georgia_data(
            data_root=args.data_root, processed_root=args.processed_root, seed=seed
        )
        assert (len(train), len(val), len(test), len(codes)) == (8260, 1038, 1046, 30)
        model_options = {}
        epochs, batch_size, eval_batch_size = args.epochs or 250, args.batch_size, 8
        validate_every = 5

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        val, batch_size=eval_batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test, batch_size=eval_batch_size, shuffle=False, num_workers=0
    )
    model = ResNet1D18(dataset=samples, **model_options).to(device)
    assert list(model.label_processor.label_vocab) == list(codes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_auc, best_state, bad_checks = -1.0, None, 0
    first_weight = model.fc2.weight.detach().clone() if args.synthetic else None
    print("Mode:", "synthetic demo" if args.synthetic else "G12EC", "device:", device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_examples = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(signal=batch["signal"].to(device),
                           labels=batch["labels"].to(device))
            output["loss"].backward()
            if args.synthetic:
                assert torch.isfinite(output["loss"])
                assert model.backbone.stem[0].weight.grad.abs().sum() > 0
            optimizer.step()
            if args.synthetic and first_weight is not None:
                assert not torch.equal(first_weight, model.fc2.weight)
                first_weight = None
            batch_size_now = batch["labels"].shape[0]
            total_loss += float(output["loss"].detach()) * batch_size_now
            total_examples += batch_size_now
        print("Epoch", epoch, "training loss", round(total_loss / total_examples, 5))

        if epoch % validate_every == 0 or epoch == epochs:
            val_prob, val_labels, _ = predict_dataset(model, val_loader, codes, device)
            val_auc, _ = evaluate_auc(
                val_labels, aggregate_predictions(val_prob)["Max"].numpy(), codes
            )
            print("Validation Max Macro ROC-AUC:", round(val_auc, 5))
            if val_auc > best_auc:
                best_auc, bad_checks = val_auc, 0
                best_state = {name: value.detach().cpu().clone()
                              for name, value in model.state_dict().items()}
                print("Best model: epoch", epoch)
            else:
                bad_checks += 1
            if bad_checks >= 5:
                print("Early stopping at epoch", epoch)
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    test_probabilities, test_labels, test_ids = predict_dataset(
        model, test_loader, codes, device
    )
    predictions = aggregate_predictions(test_probabilities)
    metrics = {name: evaluate_auc(test_labels, values.numpy(), codes)
               for name, values in predictions.items()}
    results = pd.DataFrame({
        "Aggregation": list(metrics),
        "Macro ROC-AUC": [metrics[name][0] for name in metrics],
    })
    results["Delta vs. Max"] = (
        results["Macro ROC-AUC"] - results.loc[0, "Macro ROC-AUC"]
    )
    print(
        "Test windows:", tuple(test_probabilities.shape), "record IDs:", len(test_ids)
    )
    print(
        "Synthetic demo results (not G12EC):"
        if args.synthetic else "G12EC Test results:"
    )
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
