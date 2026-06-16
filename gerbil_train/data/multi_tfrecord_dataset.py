"""Multi-class TFRecord dataset for GwEN.

The abstract base class lives in :mod:`gerbil_train.data.tfrecord_dataset`.
"""

from __future__ import annotations

from tfrecord import example_pb2

from gerbil_train.data.tfrecord_dataset import TFRecordDataset

__all__ = ["MultiTFRecordDataset"]


class MultiTFRecordDataset(TFRecordDataset):
    """TFRecord dataset for multi-class classification."""

    @staticmethod
    def _extract_target(example: example_pb2.Example) -> int:
        target_feature = example.features.feature.get("target")
        if target_feature is None:
            raise ValueError("Missing target feature in TFRecord example")
        if target_feature.float_list.value:
            return int(target_feature.float_list.value[0])
        raise ValueError("Target feature exists but has no values")
