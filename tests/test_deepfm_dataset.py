from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from gerbil_train.data.deepfm_dataset import DeepFMDataset


class DeepFMDatasetTests(unittest.TestCase):
    """Unit tests for the DeepFM ml-1m dataset pipeline."""

    def test_deepfm_dataset_builds_richer_sparse_fields(self) -> None:
        """Verify that DeepFMDataset emits the expected sparse-field width."""
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "ratings.dat").write_text(
                "1::10::4::978300760\n1::20::5::978302109\n2::30::3::978301968\n",
                encoding="latin-1",
            )
            (tmp_path / "users.dat").write_text(
                "1::F::1::10::48067\n2::M::18::20::70072\n",
                encoding="latin-1",
            )
            (tmp_path / "movies.dat").write_text(
                "10::Toy Story (1995)::Animation|Children's|Comedy\n"
                "20::Jumanji (1995)::Adventure|Children's|Fantasy\n"
                "30::Heat (1995)::Action|Crime|Thriller\n",
                encoding="latin-1",
            )

            dataset = DeepFMDataset(tmp_path / "ratings.dat", split="full")
            sample = dataset[0]

            self.assertEqual(tuple(sample["sparse_features"].shape), (7,))
            self.assertEqual(int(sample["sparse_features"][0]), 1)
            self.assertEqual(int(sample["sparse_features"][1]), 10)


if __name__ == "__main__":
    unittest.main()
