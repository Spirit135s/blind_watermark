import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_image_rgb(path: Path) -> np.ndarray:
    try:
        image = Image.open(path).convert("RGB")
    except OSError as exc:
        raise FileNotFoundError(f"Could not read image: {path}") from exc
    image_array = np.array(image)
    if image_array.size == 0:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image_array


def rgb_to_yuv(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YUV)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float(np.mean(diff * diff))


def psnr(mse_value: float) -> float:
    if mse_value == 0:
        return float("inf")
    return float(10.0 * np.log10((255.0 * 255.0) / mse_value))


def per_channel_mse(diff: np.ndarray) -> np.ndarray:
    return np.mean(diff.astype(np.float32) ** 2, axis=(0, 1))


def stretch_channel_diff(channel_diff: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """
    Stretch a single-channel absolute diff image into a visible 0-255 range.

    Uses a high percentile so a few extreme pixels do not flatten the image.
    """
    if channel_diff.size == 0:
        return np.zeros_like(channel_diff, dtype=np.uint8)

    positive = channel_diff[channel_diff > 0]
    if positive.size == 0:
        return np.zeros_like(channel_diff, dtype=np.uint8)

    upper = float(np.percentile(positive, percentile))
    if upper <= 0:
        upper = float(positive.max())
    if upper <= 0:
        return np.zeros_like(channel_diff, dtype=np.uint8)

    stretched = channel_diff.astype(np.float32) * (255.0 / upper)
    return np.clip(stretched, 0, 255).astype(np.uint8)


def save_channel_image(channel_image: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(channel_image.astype(np.uint8)).save(output_path)


def save_channel_diff(diff: np.ndarray, output_path: Path, channel: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    channel_diff = stretch_channel_diff(diff[:, :, channel])
    Image.fromarray(channel_diff).save(output_path)


def compare_images(
    original_path: Path,
    embedded_path: Path,
    original_outputs: tuple[Path | None, Path | None, Path | None] | None = None,
    diff_outputs: tuple[Path | None, Path | None, Path | None] | None = None,
) -> None:
    original_rgb = load_image_rgb(original_path)
    embedded_rgb = load_image_rgb(embedded_path)

    if original_rgb.shape != embedded_rgb.shape:
        raise ValueError(
            "Image shapes do not match: "
            f"{original_rgb.shape} vs {embedded_rgb.shape}. "
            "Compare images with the same resolution."
        )

    original_yuv = rgb_to_yuv(original_rgb)
    embedded_yuv = rgb_to_yuv(embedded_rgb)

    abs_diff = np.abs(original_yuv.astype(np.int16) - embedded_yuv.astype(np.int16))
    changed_mask = np.any(abs_diff > 0, axis=2)

    total_pixels = int(changed_mask.size)
    changed_pixels = int(np.count_nonzero(changed_mask))
    changed_ratio = changed_pixels / total_pixels if total_pixels else 0.0

    max_diff = int(abs_diff.max())
    mean_abs_diff = float(abs_diff.mean())
    mse_value = mse(original_yuv, embedded_yuv)
    psnr_value = psnr(mse_value)
    channel_labels = ("Y", "U", "V")
    channel_changed_counts = [int(np.count_nonzero(abs_diff[:, :, channel])) for channel in range(3)]
    channel_max_diff = [int(abs_diff[:, :, channel].max()) for channel in range(3)]
    channel_mean_abs_diff = [float(abs_diff[:, :, channel].mean()) for channel in range(3)]
    channel_mse = per_channel_mse(abs_diff)
    channel_psnr = [psnr(float(value)) for value in channel_mse]

    print(f"Original: {original_path}")
    print(f"Embedded:  {embedded_path}")
    print(f"Shape:     {original_rgb.shape}")
    print("Space:     YUV")
    print(f"Changed pixels: {changed_pixels} / {total_pixels} ({changed_ratio:.6%})")
    print(f"Max abs diff:   {max_diff}")
    print(f"Mean abs diff:  {mean_abs_diff:.6f}")
    print(f"MSE:            {mse_value:.6f}")
    print(f"PSNR:           {psnr_value:.4f} dB")

    print("Per-channel YUV diff statistics:")
    for index, label in enumerate(channel_labels):
        print(f"{label} channel:")
        print(f"  changed pixels: {channel_changed_counts[index]}")
        print(f"  max abs diff:   {channel_max_diff[index]}")
        print(f"  mean abs diff:  {channel_mean_abs_diff[index]:.6f}")
        print(f"  MSE:            {float(channel_mse[index]):.6f}")
        print(f"  PSNR:           {channel_psnr[index]:.4f} dB")

    if original_outputs is not None:
        for index, output_path in enumerate(original_outputs):
            if output_path is not None:
                save_channel_image(original_yuv[:, :, index], output_path)
                print(f"{channel_labels[index]} original channel saved to: {output_path}")

    if diff_outputs is not None:
        for index, output_path in enumerate(diff_outputs):
            if output_path is not None:
                save_channel_diff(abs_diff, output_path, index)
                print(f"{channel_labels[index]} diff image saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two images in YUV space and report per-channel differences."
    )
    parser.add_argument(
        "--original",
        type=Path,
        default=Path("examples/pic/ori_img.jpeg"),
        help="Path to the original image.",
    )
    parser.add_argument(
        "--embedded",
        type=Path,
        default=Path("examples/output/embedded.png"),
        help="Path to the embedded/watermarked image.",
    )
    parser.add_argument(
        "--diff-output-prefix",
        type=Path,
        default=Path("examples/output/diff_yuv"),
        help="Prefix for saving channel diff images. Files will be suffixed with _y.png, _u.png, _v.png.",
    )
    parser.add_argument(
        "--original-output-prefix",
        type=Path,
        default=Path("examples/output/original_yuv"),
        help="Prefix for saving original channel images. Files will be suffixed with _y.png, _u.png, _v.png.",
    )
    parser.add_argument(
        "--no-diff-output",
        action="store_true",
        help="Do not save diff images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    original_outputs = (
        Path(f"{args.original_output_prefix}_y.png"),
        Path(f"{args.original_output_prefix}_u.png"),
        Path(f"{args.original_output_prefix}_v.png"),
    )
    if args.no_diff_output:
        diff_outputs = None
    else:
        diff_outputs = (
            Path(f"{args.diff_output_prefix}_y.png"),
            Path(f"{args.diff_output_prefix}_u.png"),
            Path(f"{args.diff_output_prefix}_v.png"),
        )
    compare_images(args.original, args.embedded, original_outputs, diff_outputs)


if __name__ == "__main__":
    main()
