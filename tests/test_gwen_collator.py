from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from gerbil_train.data.tfrecord_dataset import BatchCollator, collect_tfrecord_part_files


class BatchCollatorTests(unittest.TestCase):
    """Tests for the GWEN batch collation logic."""

    def setUp(self):
        self.field_names = ["user_age", "user_gender", "movie_genres"]
        self.collator = BatchCollator(self.field_names)

    def test_collate_single_sample(self) -> None:
        """Single sample is collated correctly with offset=0."""
        batch = [
            {
                "targets": 5,
                "field_indices": {"user_age": [2], "user_gender": [1], "movie_genres": [3, 7]},
                "field_values": {"user_age": [1.0], "user_gender": [1.0], "movie_genres": [1.0, 0.5]},
            }
        ]
        result = self.collator(batch)

        self.assertEqual(result["targets"].tolist(), [5])
        self.assertEqual(result["targets"].dtype, torch.long)

        for field in self.field_names:
            bag = result["feature_bags"][field]
            self.assertIn("indices", bag)
            self.assertIn("offsets", bag)
            self.assertIn("weights", bag)
            self.assertEqual(bag["offsets"].tolist(), [0])
            self.assertEqual(bag["indices"].size(0), bag["weights"].size(0))

        # user_gender has 1 token
        self.assertEqual(result["feature_bags"]["user_age"]["indices"].tolist(), [2])
        self.assertEqual(result["feature_bags"]["movie_genres"]["indices"].tolist(), [3, 7])
        self.assertEqual(result["feature_bags"]["movie_genres"]["weights"].tolist(), [1.0, 0.5])

    def test_collate_two_samples(self) -> None:
        """Two samples are collated with correct offset boundaries."""
        batch = [
            {
                "targets": 1,
                "field_indices": {"user_age": [0], "user_gender": [1], "movie_genres": [2]},
                "field_values": {"user_age": [1.0], "user_gender": [1.0], "movie_genres": [1.0]},
            },
            {
                "targets": 2,
                "field_indices": {"user_age": [3, 4], "user_gender": [0], "movie_genres": []},
                "field_values": {"user_age": [1.0, 1.0], "user_gender": [1.0], "movie_genres": []},
            },
        ]
        result = self.collator(batch)

        self.assertEqual(result["targets"].tolist(), [1, 2])
        self.assertEqual(result["feature_bags"]["user_age"]["offsets"].tolist(), [0, 1])
        self.assertEqual(result["feature_bags"]["user_age"]["indices"].tolist(), [0, 3, 4])
        self.assertEqual(result["feature_bags"]["user_gender"]["offsets"].tolist(), [0, 1])
        self.assertEqual(result["feature_bags"]["user_gender"]["indices"].tolist(), [1, 0])

    def test_collate_empty_field(self) -> None:
        """A field with no tokens produces an empty indices tensor but valid offsets."""
        batch = [
            {
                "targets": 0,
                "field_indices": {"user_age": [], "user_gender": [], "movie_genres": []},
                "field_values": {"user_age": [], "user_gender": [], "movie_genres": []},
            }
        ]
        result = self.collator(batch)
        for field in self.field_names:
            self.assertEqual(result["feature_bags"][field]["indices"].tolist(), [])
            self.assertEqual(result["feature_bags"][field]["weights"].tolist(), [])

    def test_collate_three_samples_mixed_lengths(self) -> None:
        """Multiple samples with varying field lengths produce correct offsets."""
        collator = BatchCollator(["user_age"])
        batch = [
            {"targets": 10, "field_indices": {"user_age": [1]}, "field_values": {"user_age": [1.0]}},
            {"targets": 20, "field_indices": {"user_age": [2, 3]}, "field_values": {"user_age": [1.0, 1.0]}},
            {"targets": 30, "field_indices": {"user_age": [4, 5, 6]}, "field_values": {"user_age": [1.0, 1.0, 1.0]}},
        ]
        result = collator(batch)
        self.assertEqual(result["targets"].tolist(), [10, 20, 30])
        self.assertEqual(result["feature_bags"]["user_age"]["offsets"].tolist(), [0, 1, 3])
        self.assertEqual(result["feature_bags"]["user_age"]["indices"].tolist(), [1, 2, 3, 4, 5, 6])


class CollectTfrecordPartFilesTests(unittest.TestCase):
    """Tests for TFRecord file collection."""

    def test_collect_empty_directory_raises(self) -> None:
        """raise FileNotFoundError when directory has no part files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                collect_tfrecord_part_files(tmpdir)

    def test_collect_with_part_files(self) -> None:
        """Collect only part files matching the pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "part-r-00000").touch()
            (d / "part-r-00001").touch()
            (d / "other_file.txt").touch()
            (d / ".hidden").touch()

            files = collect_tfrecord_part_files(tmpdir)
            self.assertEqual(len(files), 2)
            self.assertTrue(all(f.name.startswith("part-r-") for f in files))

    def test_collect_sorted(self) -> None:
        """Returned files are sorted alphabetically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "part-r-00002").touch()
            (d / "part-r-00000").touch()
            (d / "part-r-00001").touch()

            files = collect_tfrecord_part_files(tmpdir)
            names = [f.name for f in files]
            self.assertEqual(names, ["part-r-00000", "part-r-00001", "part-r-00002"])

    def test_collect_excludes_crc(self) -> None:
        """Files ending with .crc are excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "part-r-00000").touch()
            (d / ".part-r-00000.crc").touch()

            files = collect_tfrecord_part_files(tmpdir)
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].name, "part-r-00000")


if __name__ == "__main__":
    unittest.main()
