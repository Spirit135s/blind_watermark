import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: Path) -> np.ndarray:
    try:
        image = Image.open(path).convert("RGB")
    except OSError as exc:
        raise FileNotFoundError(f"Could not read image: {path}") from exc
    image_array = np.array(image)
    if image_array.size == 0:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image_array


def mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float(np.mean(diff * diff))


def psnr(mse_value: float) -> float:
    if mse_value == 0:
        return float("inf")
    return float(10.0 * np.log10((255.0 * 255.0) / mse_value))


def save_diff_visualization(diff: np.ndarray, output_path: Path) -> None:
    if diff.ndim == 3:
        diff_gray = np.max(diff, axis=2)
    else:
        diff_gray = diff

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if diff_gray.max() == 0:
        normalized = np.zeros_like(diff_gray, dtype=np.uint8)
    else:
        normalized = (diff_gray.astype(np.float32) / diff_gray.max() * 255.0).astype(np.uint8)

    heatmap = jet_colormap(normalized)
    diff_image = Image.fromarray(heatmap, mode="RGB")
    diff_image.save(output_path)


def jet_colormap(gray: np.ndarray) -> np.ndarray:
    """Map a grayscale image to a Jet-style RGB heatmap."""
    x = gray.astype(np.float32) / 255.0

    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)

    heatmap = np.stack([r, g, b], axis=-1)
    return (heatmap * 255.0).astype(np.uint8)


def compare_images(original_path: Path, embedded_path: Path, diff_output: Path | None = None) -> None:
    original = load_image(original_path)
    embedded = load_image(embedded_path)

    if original.shape != embedded.shape:
        raise ValueError(
            "Image shapes do not match: "
            f"{original.shape} vs {embedded.shape}. "
            "Compare images with the same resolution."
        )

    abs_diff = np.abs(original.astype(np.int16) - embedded.astype(np.int16))
    changed_mask = np.any(abs_diff > 0, axis=2) if abs_diff.ndim == 3 else abs_diff > 0

    total_pixels = int(changed_mask.size)
    changed_pixels = int(np.count_nonzero(changed_mask))
    changed_ratio = changed_pixels / total_pixels if total_pixels else 0.0

    channel_changed_counts = []
    if abs_diff.ndim == 3:
        for channel in range(abs_diff.shape[2]):
            channel_changed_counts.append(int(np.count_nonzero(abs_diff[:, :, channel])))
    else:
        channel_changed_counts.append(int(np.count_nonzero(abs_diff)))

    max_diff = int(abs_diff.max())
    mean_abs_diff = float(abs_diff.mean())
    mse_value = mse(original, embedded)
    psnr_value = psnr(mse_value)

    print(f"Original: {original_path}")
    print(f"Embedded:  {embedded_path}")
    print(f"Shape:     {original.shape}")
    print(f"Changed pixels: {changed_pixels} / {total_pixels} ({changed_ratio:.6%})")
    print(f"Max abs diff:   {max_diff}")
    print(f"Mean abs diff:  {mean_abs_diff:.6f}")
    print(f"MSE:            {mse_value:.6f}")
    print(f"PSNR:           {psnr_value:.4f} dB")

    if abs_diff.ndim == 3:
        for index, count in enumerate(channel_changed_counts):
            print(f"Channel {index} changed pixels: {count}")
    else:
        print(f"Changed values: {channel_changed_counts[0]}")

    if diff_output is not None:
        save_diff_visualization(abs_diff, diff_output)
        print(f"Diff image saved to: {diff_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two images pixel by pixel and report the difference."
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
        "--diff-output",
        type=Path,
        default=Path("examples/output/diff.png"),
        help="Where to save a visual diff heatmap. Use an empty value to skip saving.",
    )
    parser.add_argument(
        "--no-diff-output",
        action="store_true",
        help="Do not save a diff heatmap.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diff_output = None if args.no_diff_output else args.diff_output
    compare_images(args.original, args.embedded, diff_output)


if __name__ == "__main__":
    main()
