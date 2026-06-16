"""TFRecord dataset for GwEN binary CTR training.

The TFRecord format for binary CTR data uses ``target`` as a float
(0.0 or 1.0) instead of a multi-class index.
"""

from __future__ import annotations

from tfrecord import example_pb2

from gerbil_train.data.tfrecord_dataset import TFRecordDataset

__all__ = ["BinaryTFRecordDataset"]


class BinaryTFRecordDataset(TFRecordDataset):
    """TFRecord dataset for binary CTR classification."""

    @staticmethod
    def _extract_target(example: example_pb2.Example) -> float:
        target_feature = example.features.feature.get("target")
        if target_feature is None:
            raise ValueError("Missing target feature in TFRecord example")
        if target_feature.float_list.value:
            rating = float(target_feature.float_list.value[0])
            return 1.0 if rating > 3.0 else 0.0
        raise ValueError("Target feature exists but has no values")
