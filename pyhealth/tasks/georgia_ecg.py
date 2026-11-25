"""
georgia dataset ecg preprocessing for the CS598 project

Jaspal Bainiwal netid: jaspalb2
Anthony Wu netid: adwu3
Pengshao Ye netid: py12
Paper: In-depth Benchmarking of Deep Neural Network Architectures
for ECG Diagnosis
https://github.com/seitalab/dnn_ecg_comparison
"""
import json
from pathlib import Path
import numpy as np
import wfdb
import torch
from pyhealth.tasks.base_task import BaseTask
from torch.utils.data import Dataset, Subset
from pyhealth.datasets import CardiologyDataset

def load_normalized_ecg(
    record_path: str | Path,
    mean: np.ndarray,
    std: np.ndarray,
    lead_names: list[str],
) -> np.ndarray:
    """read ecg in mV and normalize each lead using training stats

    Args:
        record_path: Path to the recording without .hea or .mat
        mean: Mean for each lead from the training set
        std: standard dev. for each lead from the training set
        lead_names: Lead order used when computing the statistics

    Returns:
        array with shape (12, number of samples)
    """
    record = wfdb.rdrecord(str(record_path))

    if record.fs != 500:
        raise ValueError("Expect 500 Hz recording")
    if any(unit != "mV" for unit in record.units):
        raise ValueError("Expected signal values in mv format")
    if record.n_sig != 12 or record.sig_name != list(lead_names):
        raise ValueError("Unexpected ECG leads or lead order")

    mean = np.asarray(mean)
    std = np.asarray(std)

    if mean.shape != (12,) or std.shape != (12,):
        raise ValueError("expect 12 means and 12 standard deviations")
    #prevent training loss with nan vals
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("normalize statistics must be finite")

    if (std <= 0).any():
        raise ValueError("Standard deviations must be positive")

    #put leads first and replace missing values before normalization
    signal = record.p_signal.T.copy()
    signal[np.isnan(signal)] = 0

    if not np.isfinite(signal).all():
        raise ValueError("signal contains infinite vals")

    #pad shorter recordings to 10 seconds before normalization 
    target_length = 5000
    signal_length = signal.shape[1]

    if signal_length > target_length:
        raise ValueError("Recording exceeds 10 seconds")

    if signal_length < target_length:
        padding = target_length - signal_length
        signal = np.pad(signal, ((0, 0), (padding, 0)))
    #end padding for recordings shorter than 10 seconds
    #use training statistics for training validation and test recordings
    signal = (signal - mean[:, None]) / std[:, None]

    return signal.astype(np.float32)

def random_window(
    signal: np.ndarray,
    rng: np.random.Generator,
    window_size: int = 1250,
) -> np.ndarray:
    """during training pick a random section of the ecg for training.
       the window is 500 hz 10 seconds so we take 2.5 seconds 
    """
    if window_size <= 0:
        raise ValueError("window size must be positive")

    if signal.shape[1] < window_size:
        raise ValueError("recording is shorter than the window")

    #include the last valid starting position
    start = int(rng.integers(0, signal.shape[1] - window_size + 1))

    end = start + window_size
    window = signal[:, start:end].copy()

    return window

def eval_window(
    signal: np.ndarray,
    window_size: int = 1250,
) -> np.ndarray:
    """
    split the ecg into overlapping windows for evaluation
    """
    if window_size < 2 or window_size % 2 != 0:
        raise ValueError("window size must be a positive even number")

    if signal.shape[1] < window_size:
        raise ValueError("recording is shorter than the window")

    step = window_size // 2

    #match the padding used by the authors
    padding = step - signal.shape[1] % step
    padded = np.pad(signal, ((0, 0), (0, padding)))

    windows = []
    stop = padded.shape[1] - window_size + 1

    for start in range(0, stop, step):
        end = start + window_size
        window = padded[:, start:end]
        windows.append(window)

    result = np.stack(windows)

    return result

class GeorgiaECGClassification(BaseTask):
    """
    load georgia ecg signals
    also load diagnosis labels
    """

    #task_name = "GeorgiaECGClassification"
    #updated for caching purposes related to padding issues
    task_name = "GeorgiaECGClassification_leftpad_v2"
    input_schema = {"signal": "tensor"}

    def __init__(self, codes, mean, std, lead_names):
        super().__init__()

        self.codes = list(codes)
        if len(self.codes) != 30 or len(set(self.codes)) != 30:
            raise ValueError("expected 30 unique codes")

        self.mean = np.asarray(mean).copy()
        self.std = np.asarray(std).copy()
        self.lead_names = list(lead_names)

        #use the same label order for each record
        self.output_schema = {
            "labels": ("multilabel", {"labels": self.codes})
        }

    def __call__(self, patient):
        samples = []

        for event in patient.get_events():
            if event.source != "georgia":
                continue

            record_path = Path(event.signal_file).with_suffix("")
            signal = load_normalized_ecg(
                record_path,
                self.mean,
                self.std,
                self.lead_names,
            )

            dx = str(event.dx or "")
            dx = dx.replace(";", ",")
            diagnoses = set()

            for code in dx.split(","):
                code = code.strip()
                if code:
                    diagnoses.add(code)

            #keep only codes from our list of 30
            labels = []
            for code in self.codes:
                if code in diagnoses:
                    labels.append(code)

            sample = {
                "patient_id": patient.patient_id,
                "record_id": event.record_id,
                "source": event.source,
                "signal": torch.from_numpy(signal),
                "labels": labels,
            }
            samples.append(sample)

        return samples

class GeorgiaTrainingDataset(Dataset):
    """
    select a random ecg window when loading a training sample
    """

    def __init__(self, samples, seed=1):
        self.samples = samples
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = dict(self.samples[index])
        signal = np.asarray(sample["signal"])

        window = random_window(signal, self.rng)
        sample["signal"] = torch.from_numpy(window)

        return sample

class GeorgiaEvaluationDataset(Dataset):
    """
    load all evaluation windows for one recording
    """

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = dict(self.samples[index])
        signal = np.asarray(sample["signal"])

        windows = eval_window(signal)
        sample["signal"] = torch.from_numpy(windows)

        return sample

def load_georgia_data(data_root, processed_root, seed=1):
    """
    load the saved splits and prepare training and evaluation datasets
    """
    processed_root = Path(processed_root)

    #seed control window selection the saved split stays fixed
    stats_path = processed_root / "normalization_seed1.npz"
    label_path = processed_root / "label_order.json"
    split_path = processed_root / "split_seed1.npz"


    codes = json.loads(label_path.read_text())

    with np.load(split_path) as saved:
        record_ids = saved["record_ids"].copy()
        split_indices = {}

        for name in ["train", "val", "test"]:
            split_indices[name] = saved[name + "_idx"].copy()

    #each recording must appear exactly once in three splits
    all_indices = np.concatenate(list(split_indices.values()))
    expected_indices = np.arange(len(record_ids))

    if not np.array_equal(np.sort(all_indices), expected_indices):
        raise ValueError("saved splits overlap or have missing recordings")

    if len(set(record_ids)) != len(record_ids):
        raise ValueError("saved recording ids are not unique")

    #the saved stats must belong to this split
    with np.load(stats_path) as stats:
        same_training_split = np.array_equal(
            stats["train_idx"],
            split_indices["train"],
        )

        if not same_training_split:
            raise ValueError("normalization and training split not match")

        task = GeorgiaECGClassification(
            codes,
            stats["mean"],
            stats["std"],
            stats["lead_names"].tolist(),
        )

    dataset = CardiologyDataset(
        root=str(Path(data_root).resolve()),
        sources=["georgia"],
        num_workers=1,
        dev=False,
    )
    samples = dataset.set_task(task)

    #pyhealth may return recordings in a different order than the saved file
    sample_indices = {}

    for index in range(len(samples)):
        record_id = samples[index]["record_id"]

        if record_id in sample_indices:
            raise ValueError(f"Duplicate recording - {record_id}")

        sample_indices[record_id] = index

    saved_ids = set(record_ids)
    loaded_ids = set(sample_indices)

    if loaded_ids != saved_ids:
        raise ValueError("dataset recordings do not match saved split")

    #translate the saved split positions into pyhealth sample positions
    subsets = {}

    for name in ["train", "val", "test"]:
        indices = []

        for saved_index in split_indices[name]:
            record_id = record_ids[saved_index]
            sample_index = sample_indices[record_id]
            indices.append(sample_index)

        subsets[name] = Subset(samples, indices)
    test = GeorgiaEvaluationDataset(subsets["test"])
    train = GeorgiaTrainingDataset(subsets["train"], seed=seed)
    val = GeorgiaEvaluationDataset(subsets["val"])

    return train, val, test, codes