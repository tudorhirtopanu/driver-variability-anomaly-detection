"""Shared window-dataset helpers for false-positive analysis experiments."""

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset


class ReconWindowDataset(Dataset):
    """Return fixed-length reconstruction windows from a single sequence."""

    def __init__(self, data: np.ndarray, window_size: int, stride: int):
        super().__init__()
        data_t = torch.as_tensor(data, dtype=torch.float32)
        if data_t.ndim != 2:
            raise ValueError(f"Expected (N, D), got {tuple(data_t.shape)}")
        self.data = data_t
        self.num_rows, _ = data_t.shape
        self.window_size = window_size
        self.stride = stride
        if self.stride < 1:
            raise ValueError("stride must be >= 1")

        self.num_windows = (self.num_rows - self.window_size) // self.stride + 1
        if self.num_windows <= 0:
            raise ValueError(f"Not enough points: N={self.num_rows}, T={self.window_size}")

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx: int):
        start = idx * self.stride
        end = start + self.window_size
        return self.data[start:end]


def build_window_dataset(
    seqs: List[np.ndarray],
    window_size: int,
    stride: int,
) -> Dataset:
    datasets = [ReconWindowDataset(seq, window_size=window_size, stride=stride) for seq in seqs]
    return ConcatDataset(datasets)


def build_window_dataset_with_ids(
    seqs: List[np.ndarray],
    ids_per_sequence: List[str],
    window_size: int,
    stride: int,
) -> Tuple[Dataset, np.ndarray]:
    datasets = []
    ids_per_window = []
    for seq, item_id in zip(seqs, ids_per_sequence):
        dataset = ReconWindowDataset(seq, window_size=window_size, stride=stride)
        datasets.append(dataset)
        ids_per_window.extend([item_id] * len(dataset))
    return ConcatDataset(datasets), np.array(ids_per_window, dtype=object)


class ForecastWindowDataset(Dataset):
    """Return context windows paired with the next-step forecasting target."""

    def __init__(self, data: np.ndarray, window_size: int, stride: int):
        super().__init__()
        data_t = torch.as_tensor(data, dtype=torch.float32)
        if data_t.ndim != 2:
            raise ValueError(f"Expected (N, D), got {tuple(data_t.shape)}")

        self.data = data_t
        self.num_rows, _ = data_t.shape
        self.window_size = window_size
        self.stride = stride
        if self.stride < 1:
            raise ValueError("stride must be >= 1")

        self.num_windows = (self.num_rows - self.window_size - 1) // self.stride + 1
        if self.num_windows <= 0:
            raise ValueError(
                f"Not enough points for forecasting: N={self.num_rows}, T={self.window_size}"
            )

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx: int):
        start = idx * self.stride
        end = start + self.window_size
        target_idx = end
        return self.data[start:end], self.data[target_idx]


def build_forecast_dataset(
    seqs: List[np.ndarray],
    window_size: int,
    stride: int,
) -> Dataset:
    datasets = [ForecastWindowDataset(seq, window_size=window_size, stride=stride) for seq in seqs]
    return ConcatDataset(datasets)


def build_forecast_dataset_with_ids(
    seqs: List[np.ndarray],
    ids_per_sequence: List[str],
    window_size: int,
    stride: int,
) -> Tuple[Dataset, np.ndarray]:
    datasets = []
    ids_per_window = []
    for seq, item_id in zip(seqs, ids_per_sequence):
        dataset = ForecastWindowDataset(seq, window_size=window_size, stride=stride)
        datasets.append(dataset)
        ids_per_window.extend([item_id] * len(dataset))
    return ConcatDataset(datasets), np.array(ids_per_window, dtype=object)
