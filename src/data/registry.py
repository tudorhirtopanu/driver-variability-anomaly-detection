"""Dataset registry used by the shared training and evaluation entrypoints.

The public scripts expose one interface for HCRL, Sonata, and OBD, but the
datasets still differ slightly in column cleaning, sorting, and split style.
This registry keeps those differences in one place so the higher-level code can
stay mostly dataset-agnostic.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import pandas as pd

from . import hcrl, obd, sonata
from . import preprocessing as common_preproc

DFHook = Optional[Callable[[pd.DataFrame], pd.DataFrame]]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    meta_cols: Sequence[str]
    ignored_features: Sequence[str]
    sort_cols: Sequence[str]
    preprocess_df: DFHook
    uses_driver_splits: bool


@dataclass(frozen=True)
class DatasetRuntime:
    spec: DatasetSpec
    module: object

    @property
    def META_COLS(self):
        return self.spec.meta_cols

    @property
    def IGNORED_FEATURES(self):
        return self.spec.ignored_features

    def parse_driver_list(self, value: Optional[str]):
        return self.module.parse_driver_list(value)

    def list_files_for_drivers(self, data_dir: Path, drivers: Sequence[str]):
        return self.module.list_files_for_drivers(data_dir, drivers)

    def load_and_clean_csv(self, path: Path, meta_cols, ignored_features):
        return common_preproc.load_and_clean_csv(
            path,
            meta_cols=meta_cols,
            ignored_features=ignored_features,
            sort_cols=self.spec.sort_cols,
            preprocess_df=self.spec.preprocess_df,
        )

    def align_columns(self, df: pd.DataFrame, expected_cols: Sequence[str], path: Path):
        return common_preproc.align_columns(df, expected_cols, path)

    def compute_train_stats(self, train_dfs):
        return common_preproc.compute_train_stats(train_dfs)

    def normalize_df(self, df: pd.DataFrame, feature_cols, means, stds):
        return common_preproc.normalize_df(df, feature_cols, means, stds)

    def load_split_dfs(
        self,
        files,
        base_cols,
        feature_cols,
        means,
        stds,
        split_name: str,
    ):
        return common_preproc.load_split_dfs(
            files,
            base_cols=base_cols,
            feature_cols=feature_cols,
            means=means,
            stds=stds,
            split_name=split_name,
            meta_cols=self.spec.meta_cols,
            ignored_features=self.spec.ignored_features,
            sort_cols=self.spec.sort_cols,
            preprocess_df=self.spec.preprocess_df,
        )

    def build_windows_from_dfs(self, dfs, lookback: int):
        return common_preproc.build_windows_from_dfs(dfs, lookback)


def get_dataset_spec(name: str) -> DatasetSpec:
    key = name.lower().strip()

    if key == "hcrl":
        return DatasetSpec(
            name="hcrl",
            meta_cols=hcrl.META_COLS,
            ignored_features=hcrl.IGNORED_FEATURES,
            sort_cols=("Time(s)",),
            preprocess_df=None,
            uses_driver_splits=True,
        )

    if key == "sonata":
        return DatasetSpec(
            name="sonata",
            meta_cols=sonata.META_COLS,
            ignored_features=sonata.IGNORED_FEATURES,
            sort_cols=("Time(s)",),
            preprocess_df=None,
            uses_driver_splits=True,
        )

    if key == "obd":
        return DatasetSpec(
            name="obd",
            meta_cols=obd.META_COLS,
            ignored_features=obd.IGNORED_FEATURES,
            sort_cols=("Time", "Time(s)"),
            # OBD files sometimes contain encoding artefacts in column names,
            # so they are normalized before schema checks or normalization.
            preprocess_df=obd._normalize_columns,
            uses_driver_splits=False,
        )

    raise ValueError(f"Unknown dataset: {name}. Choose from: hcrl, sonata, obd")


def get_dataset(name: str) -> DatasetRuntime:
    spec = get_dataset_spec(name)
    if spec.name == "hcrl":
        module = hcrl
    elif spec.name == "sonata":
        module = sonata
    elif spec.name == "obd":
        module = obd
    else:
        raise ValueError(f"Unknown dataset: {name}. Choose from: hcrl, sonata, obd")

    return DatasetRuntime(spec=spec, module=module)
