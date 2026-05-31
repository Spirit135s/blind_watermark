#!/usr/bin/env python3
# coding=utf-8
"""
Run robustness tests for the no-password course watermark demo.

The perturbations come from the original project's blind_watermark.att module.
Results are written to CSV so they can be copied into the course report.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blind_watermark import att
from simple_watermark_demo import SimpleWatermarkDemo, bits_to_text, save_bgr, text_to_bits


def bit_error_count(expected: np.ndarray, actual: np.ndarray) -> int:
    return int(np.count_nonzero(expected.astype(np.uint8) != actual.astype(np.uint8)))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float32) - b.astype(np.float32)
    mse = float(np.mean(diff * diff))
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10((255.0 * 255.0) / mse))


def jpeg_attack(img: np.ndarray, quality: int) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed.")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("JPEG decoding failed.")
    return decoded


def build_attacks(original_shape: tuple[int, int]) -> list[tuple[str, str, callable]]:
    h, w = original_shape
    return [
        ("none", "clean", lambda img: img.copy()),
        ("jpeg", "quality_95", lambda img: jpeg_attack(img, 95)),
        ("jpeg", "quality_80", lambda img: jpeg_attack(img, 80)),
        ("jpeg", "quality_60", lambda img: jpeg_attack(img, 60)),
        ("jpeg", "quality_40", lambda img: jpeg_attack(img, 40)),
        ("resize", "scale_0.75_restore", lambda img: att.resize_att(input_img=att.resize_att(input_img=img, out_shape=(round(w * 0.75), round(h * 0.75))), out_shape=(w, h))),
        ("resize", "scale_0.50_restore", lambda img: att.resize_att(input_img=att.resize_att(input_img=img, out_shape=(round(w * 0.50), round(h * 0.50))), out_shape=(w, h))),
        ("noise", "salt_pepper_0.005", lambda img: att.salt_pepper_att(input_img=img, ratio=0.005)),
        ("noise", "salt_pepper_0.010", lambda img: att.salt_pepper_att(input_img=img, ratio=0.010)),
        ("noise", "salt_pepper_0.030", lambda img: att.salt_pepper_att(input_img=img, ratio=0.030)),
        ("brightness", "ratio_0.85", lambda img: att.bright_att(input_img=img, ratio=0.85)),
        ("brightness", "ratio_1.15", lambda img: att.bright_att(input_img=img, ratio=1.15)),
        ("shelter", "ratio_0.05_n_5", lambda img: att.shelter_att(input_img=img, ratio=0.05, n=5)),
        ("shelter", "ratio_0.10_n_5", lambda img: att.shelter_att(input_img=img, ratio=0.10, n=5)),
        ("rotate", "angle_5_restore", lambda img: att.rot_att(input_img=att.rot_att(input_img=img, angle=5), angle=-5)),
        ("rotate", "angle_15_restore", lambda img: att.rot_att(input_img=att.rot_att(input_img=img, angle=15), angle=-15)),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robustness test for the no-password DWT-DCT-SVD demo.")
    parser.add_argument("--input", type=Path, default=Path("examples/pic/ori_img.jpeg"))
    parser.add_argument("--output-dir", type=Path, default=Path("course_project/robustness_output"))
    parser.add_argument("--watermark", default="SignalSystem")
    parser.add_argument("--d1", type=float, default=36)
    parser.add_argument("--d2", type=float, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    img = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.input}")

    output_dir = args.output_dir
    attacks_dir = output_dir / "attacked_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    attacks_dir.mkdir(parents=True, exist_ok=True)

    wm_bits = text_to_bits(args.watermark)
    demo = SimpleWatermarkDemo(d1=args.d1, d2=args.d2)
    embedded = demo.embed(img, wm_bits)
    save_bgr(output_dir / "watermarked.png", embedded)

    rows = []
    for attack_type, attack_param, attack_fn in build_attacks(img.shape[:2]):
        attacked = attack_fn(embedded)
        attacked = np.clip(attacked, 0, 255).astype(np.uint8)
        attacked_path = attacks_dir / f"{attack_type}_{attack_param}.png"
        save_bgr(attacked_path, attacked)

        extracted_bits = demo.extract_bits(attacked, wm_size=wm_bits.size)
        extracted_text = bits_to_text(extracted_bits)
        errors = bit_error_count(wm_bits, extracted_bits)
        ber = errors / wm_bits.size
        rows.append(
            {
                "attack_type": attack_type,
                "attack_param": attack_param,
                "bit_errors": errors,
                "bit_error_rate": f"{ber:.6f}",
                "text_exact": str(extracted_text == args.watermark),
                "extracted_text": extracted_text,
                "psnr_vs_watermarked": f"{psnr(embedded, attacked):.4f}",
                "attacked_image": str(attacked_path),
            }
        )

    csv_path = output_dir / "robustness_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "attack_type",
                "attack_param",
                "bit_errors",
                "bit_error_rate",
                "text_exact",
                "extracted_text",
                "psnr_vs_watermarked",
                "attacked_image",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Watermark: {args.watermark}")
    print(f"Results: {csv_path}")
    for row in rows:
        print(
            f"{row['attack_type']:10s} {row['attack_param']:20s} "
            f"errors={row['bit_errors']:>3} ber={row['bit_error_rate']} exact={row['text_exact']}"
        )


if __name__ == "__main__":
    main()
