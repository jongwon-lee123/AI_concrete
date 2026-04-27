import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
import json

from flow import (
    _normalize_path_string,
    auto_detect_image_dir,
    classify_ks,
    collect_image_paths,
    resolve_input_dir,
    load_input_json,
    save_result_json,
    find_path_candidates,
)


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

    def test_load_input_json(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td, "input.json")
            p.write_text(json.dumps({"image_dir": "x", "brass_od_mm": 250}), encoding="utf-8")
            data = load_input_json(str(p))
            self.assertEqual(data["image_dir"], "x")

    def test_save_result_json(self) -> None:
        with TemporaryDirectory() as td:
            out = Path(td, "out", "result.json")
            save_result_json(
                path=str(out),
                image_paths=["a.jpg", "b.jpg"],
                composite_path="out/composite.png",
                csv_path="out/result.csv",
                row_count=2,
            )
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["image_count"], 2)
            self.assertEqual(payload["row_count"], 2)

    def test_find_path_candidates_existing_path(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td, "sample.txt")
            p.write_text("x", encoding="utf-8")
            found = find_path_candidates(str(p))
            self.assertEqual(found[0], str(p.resolve()))

    def test_auto_detect_image_dir(self) -> None:
        with TemporaryDirectory() as td:
            folder = Path(td, "images")
            folder.mkdir(parents=True, exist_ok=True)
            Path(folder, "a.jpg").write_text("x", encoding="utf-8")
            Path(folder, "b.png").write_text("x", encoding="utf-8")
            detected = auto_detect_image_dir(search_roots=[Path(td)], min_images=2)
            self.assertEqual(detected, str(folder.resolve()))

    def test_resolve_input_dir_prefers_explicit(self) -> None:
        resolved = resolve_input_dir("C:/explicit/path", default_candidates=["C:/dummy"])
        self.assertEqual(resolved, "C:/explicit/path")

    def test_normalize_path_string(self) -> None:
        self.assertEqual(_normalize_path_string('"C:/a/b.jpg"'), "C:/a/b.jpg")
        self.assertEqual(_normalize_path_string("'C:/a/b.jpg'"), "C:/a/b.jpg")


if __name__ == "__main__":
    unittest.main()
