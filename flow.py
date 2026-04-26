"""Flow table image analysis utilities.

This module detects:
1) Brass plate outer circle (for scale calibration)
2) Mortar spread contour (for D, Flow calculation)

It is written as a script-friendly module and can run both in notebooks and CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import math
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd



@dataclass(frozen=True)
class FlowMeasurement:
    image_index: int
    image_path: str
    brass_outer_diameter_mm: float
    avg_diameter_mm: float
    major_axis_mm: float
    minor_axis_mm: float
    flow_mm: float
    verdict: str


def classify_ks(flow_mm: float, target_mm: float = 110.0, tol_mm: float = 5.0) -> str:
    lower = target_mm - tol_mm
    upper = target_mm + tol_mm
    if flow_mm < lower:
        return "KS 110±5 미달"
    if flow_mm <= upper:
        return "KS 범위"
    return "KS 초과"


def _ensure_image(img_bgr, image_path: str):
    if img_bgr is None:
        raise RuntimeError(f"이미지를 읽지 못했습니다: {image_path}")
    return img_bgr


def detect_brass_circle(img_bgr) -> tuple[int, int, int]:
    import cv2
    import numpy as np

    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (9, 9), 1.2)

    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(h, w) // 3,
        param1=120,
        param2=35,
        minRadius=int(min(h, w) * 0.20),
        maxRadius=int(min(h, w) * 0.49),
    )
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        # Most likely brass ring: largest radius with center near image center.
        cx0, cy0 = w / 2.0, h / 2.0
        scored = []
        for c in circles:
            x, y, r = int(c[0]), int(c[1]), int(c[2])
            center_dist = math.hypot(x - cx0, y - cy0)
            scored.append((r - center_dist * 0.20, x, y, r))
        _, x, y, r = max(scored, key=lambda t: t[0])
        return x, y, r

    # Fallback: HSV mask for brass-like color.
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([10, 15, 70]), np.array([45, 220, 255]))
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("황동 원을 검출하지 못했습니다")

    c = max(contours, key=cv2.contourArea)
    (x, y), r = cv2.minEnclosingCircle(c)
    return int(x), int(y), int(r)


def detect_mortar_contour(img_bgr, x_c: int, y_c: int, r_brass: int):
    import cv2
    import numpy as np

    # Use LAB to better separate gray mortar from yellow brass.
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Mortar candidate: darker + less yellow than brass.
    dark_mask = cv2.inRange(l, 0, 170)
    less_yellow_mask = cv2.inRange(b, 0, 150)
    mask = cv2.bitwise_and(dark_mask, less_yellow_mask)

    # Restrict analysis to the plate interior but avoid outer edge glare/scratch.
    roi = np.zeros_like(mask)
    cv2.circle(roi, (x_c, y_c), int(r_brass * 0.83), 255, -1)
    mask = cv2.bitwise_and(mask, roi)

    # Smooth mask.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("모르타르 경계를 검출하지 못했습니다")

    # Score by area + closeness to center + circularity.
    candidates: list[tuple[float, np.ndarray]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 4000:
            continue
        peri = cv2.arcLength(c, True)
        if peri == 0:
            continue
        circularity = 4 * math.pi * area / (peri * peri)
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        dist = math.hypot(cx - x_c, cy - y_c)
        score = area + circularity * 50000 - dist * 150
        candidates.append((score, c))

    if not candidates:
        return max(contours, key=cv2.contourArea)

    return max(candidates, key=lambda t: t[0])[1]


def contour_axes_px(contour) -> tuple[float, float, tuple | None]:
    import cv2

    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        (_, _), (a_px, b_px), _ = ellipse
        return float(max(a_px, b_px)), float(min(a_px, b_px)), ellipse

    x, y, w, h = cv2.boundingRect(contour)
    return float(max(w, h)), float(min(w, h)), None


def analyze_images(
    image_paths: list[str],
    brass_outer_diameter_mm: float = 250.0,
    mold_base_diameter_mm: float = 100.0,
    out_dir: str = ".",
) -> tuple["pd.DataFrame", str, str]:
    import cv2
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results: list[FlowMeasurement] = []
    annotated_paths: list[str] = []

    for idx, p in enumerate(image_paths, 1):
        img = _ensure_image(cv2.imread(p), p)

        x_c, y_c, r_brass = detect_brass_circle(img)
        mm_per_px = brass_outer_diameter_mm / (2 * r_brass)

        contour = detect_mortar_contour(img, x_c, y_c, r_brass)
        major_px, minor_px, ellipse = contour_axes_px(contour)

        major_mm = major_px * mm_per_px
        minor_mm = minor_px * mm_per_px
        avg_mm = (major_mm + minor_mm) / 2.0
        flow_mm = avg_mm - mold_base_diameter_mm

        results.append(
            FlowMeasurement(
                image_index=idx,
                image_path=p,
                brass_outer_diameter_mm=brass_outer_diameter_mm,
                avg_diameter_mm=round(avg_mm, 1),
                major_axis_mm=round(major_mm, 1),
                minor_axis_mm=round(minor_mm, 1),
                flow_mm=round(flow_mm, 1),
                verdict=classify_ks(flow_mm),
            )
        )

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ann = rgb.copy()
        cv2.circle(ann, (x_c, y_c), r_brass, (255, 0, 0), 6)
        cv2.drawContours(ann, [contour], -1, (0, 255, 0), 6)
        if ellipse is not None:
            cv2.ellipse(ann, ellipse, (255, 255, 0), 6)

        cv2.putText(
            ann,
            f"#{idx} D={avg_mm:.1f} mm, Flow={flow_mm:.1f} mm",
            (40, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (255, 255, 255),
            4,
            cv2.LINE_AA,
        )

        annotated_file = out_path / f"flow_annotated_{idx}.png"
        plt.imsave(str(annotated_file), ann)
        annotated_paths.append(str(annotated_file))

    df = pd.DataFrame(
        [
            {
                "사진": r.image_index,
                "이미지 경로": r.image_path,
                "황동 외경 가정(mm)": r.brass_outer_diameter_mm,
                "평균 확산 직경 D(mm)": r.avg_diameter_mm,
                "장축(mm)": r.major_axis_mm,
                "단축(mm)": r.minor_axis_mm,
                "플로우값 D-100(mm)": r.flow_mm,
                "판정": r.verdict,
            }
            for r in results
        ]
    )

    n = len(annotated_paths)
    cols = 2
    rows = math.ceil(n / cols)
    fig = plt.figure(figsize=(16, 8 * rows))
    for i, ap in enumerate(annotated_paths, 1):
        ax = fig.add_subplot(rows, cols, i)
        img = plt.imread(ap)
        ax.imshow(img)
        ax.set_title(
            f"Image {i}: D={df.loc[i-1, '평균 확산 직경 D(mm)']} mm / "
            f"Flow={df.loc[i-1, '플로우값 D-100(mm)']} mm"
        )
        ax.axis("off")

    plt.tight_layout()
    composite_path = out_path / "flow_test_annotated_results.png"
    plt.savefig(composite_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    csv_path = out_path / "flow_test_results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    return df, str(composite_path), str(csv_path)


def collect_image_paths(image_paths: list[str], image_dir: str | None = None) -> list[str]:
    """Collect image files from explicit paths and/or a directory."""
    supported_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    collected = [p for p in image_paths if Path(p).suffix.lower() in supported_ext]

    if image_dir:
        folder = Path(image_dir)
        if not folder.exists():
            raise RuntimeError(f"이미지 폴더가 존재하지 않습니다: {image_dir}")
        from_dir = [
            str(p)
            for p in sorted(folder.iterdir())
            if p.is_file() and p.suffix.lower() in supported_ext
        ]
        collected.extend(from_dir)

    # Keep insertion order while removing duplicates.
    deduped = list(dict.fromkeys(collected))
    if not deduped:
        raise RuntimeError("분석할 이미지가 없습니다. 경로나 폴더를 확인하세요.")
    return deduped


def load_input_json(path: str) -> dict:
    """Load input configuration from a JSON file."""
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"입력 JSON 파일이 없습니다: {path}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError("입력 JSON 형식이 잘못되었습니다. 객체(dict)여야 합니다.")
    return data


def save_result_json(
    path: str,
    image_paths: list[str],
    composite_path: str,
    csv_path: str,
    row_count: int,
) -> None:
    """Save result metadata to JSON."""
    payload = {
        "image_count": len(image_paths),
        "image_paths": image_paths,
        "composite_path": composite_path,
        "csv_path": csv_path,
        "row_count": row_count,
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flow table image analyzer")
    parser.add_argument("image_paths", nargs="*", help="Input image file paths")
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Folder path containing images (.jpg/.jpeg/.png/.bmp/.tif/.tiff)",
    )
    parser.add_argument("--brass-od-mm", type=float, default=250.0)
    parser.add_argument("--mold-base-mm", type=float, default=100.0)
    parser.add_argument("--out-dir", default=".")
    parser.add_argument(
        "--save-in-image-dir",
        action="store_true",
        help="Save outputs into image_dir folder instead of out-dir",
    )
    parser.add_argument("--input-json", default=None, help="Path to input config JSON")
    parser.add_argument(
        "--result-json",
        default=None,
        help="Optional output JSON path for saving result metadata",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information (script path, image count, output path)",
    )
    parser.add_argument(
        "--show-path",
        action="store_true",
        help="Print absolute path of this flow.py and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.show_path:
            print(Path(__file__).resolve())
            return

        # Optional JSON input config overrides CLI values.
        if args.input_json:
            cfg = load_input_json(args.input_json)
            args.image_paths = cfg.get("image_paths", args.image_paths)
            args.image_dir = cfg.get("image_dir", args.image_dir)
            args.brass_od_mm = float(cfg.get("brass_od_mm", args.brass_od_mm))
            args.mold_base_mm = float(cfg.get("mold_base_mm", args.mold_base_mm))
            args.out_dir = cfg.get("out_dir", args.out_dir)
            args.result_json = cfg.get("result_json", args.result_json)
            args.save_in_image_dir = bool(cfg.get("save_in_image_dir", args.save_in_image_dir))

        if args.save_in_image_dir and args.image_dir:
            args.out_dir = args.image_dir

        image_paths = collect_image_paths(args.image_paths, args.image_dir)
        if args.debug:
            print(f"[debug] script={Path(__file__).resolve()}")
            print(f"[debug] images={len(image_paths)}")
            print(f"[debug] out_dir={Path(args.out_dir).resolve()}")

        df, composite_path, csv_path = analyze_images(
            image_paths=image_paths,
            brass_outer_diameter_mm=args.brass_od_mm,
            mold_base_diameter_mm=args.mold_base_mm,
            out_dir=args.out_dir,
        )
        print(df)
        print(f"composite_path={composite_path}")
        print(f"csv_path={csv_path}")
        if args.result_json:
            save_result_json(
                path=args.result_json,
                image_paths=image_paths,
                composite_path=composite_path,
                csv_path=csv_path,
                row_count=len(df),
            )
            print(f"result_json={args.result_json}")
    except ModuleNotFoundError as e:
        missing = e.name or "dependency"
        raise RuntimeError(
            f"필수 라이브러리가 없습니다: {missing}. "
            "다음 명령으로 설치 후 다시 실행하세요: "
            "pip install opencv-python numpy pandas matplotlib"
        ) from e


if __name__ == "__main__":
    main()
