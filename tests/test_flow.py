import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from flow import classify_ks, collect_image_paths


class FlowRuleTests(unittest.TestCase):
    def test_classify_ks_below(self) -> None:
        self.assertEqual(classify_ks(80.0), "KS 110±5 미달")

    def test_classify_ks_in_range(self) -> None:
        self.assertEqual(classify_ks(110.0), "KS 범위")

    def test_classify_ks_above(self) -> None:
        self.assertEqual(classify_ks(123.2), "KS 초과")

    def test_collect_image_paths_from_directory(self) -> None:
        with TemporaryDirectory() as td:
            Path(td, "a.jpg").write_text("x")
            Path(td, "b.png").write_text("x")
            Path(td, "note.txt").write_text("x")
            paths = collect_image_paths([], td)
            self.assertEqual(len(paths), 2)

    def test_collect_image_paths_empty_raises(self) -> None:
        with TemporaryDirectory() as td:
            with self.assertRaises(RuntimeError):
                collect_image_paths([], td)


if __name__ == "__main__":
    unittest.main()
