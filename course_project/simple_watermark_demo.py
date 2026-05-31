#!/usr/bin/env python3
# coding=utf-8
"""
Course-project oriented DWT-DCT-SVD watermark demo.

This module intentionally removes the password/shuffle parts from the original
library so the algorithm path is easy to present:

image -> YUV -> DWT low-frequency ca -> 4x4 blocks -> DCT -> SVD
      -> singular-value modulation -> inverse transforms -> watermarked image

It also exports intermediate images for a report or slide deck.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from pywt import dwt2, idwt2


BLOCK_SHAPE = (4, 4)


def text_to_bits(text: str) -> np.ndarray:
    # 将字符串水印转成 0/1 bit 序列。后续算法只处理 bit，不关心水印原本是文本还是图片。
    data = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
    return np.unpackbits(data).astype(np.uint8)


def bits_to_text(bits: np.ndarray) -> str:
    # 将提取出的 bit 序列按 8 bit 一组重新打包成字节，再按 UTF-8 解码回字符串。
    bit_array = np.asarray(bits, dtype=np.uint8)
    if bit_array.size % 8:
        pad = 8 - bit_array.size % 8
        bit_array = np.concatenate([bit_array, np.zeros(pad, dtype=np.uint8)])
    data = np.packbits(bit_array).tobytes()
    return data.decode("utf-8", errors="replace")


def normalize_to_u8(arr: np.ndarray) -> np.ndarray:
    # DWT/DCT/SVD 中间结果通常是浮点数，范围可能不是 0-255。
    # 为了保存成图片，这里把任意矩阵线性拉伸到 uint8 灰度图。
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    min_value = float(np.min(arr))
    max_value = float(np.max(arr))
    if max_value <= min_value:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr - min_value) * (255.0 / (max_value - min_value))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def save_gray(path: Path, arr: np.ndarray) -> None:
    # 保存单通道可视化图，例如 Y 通道、DWT 低频 ca、DCT 频谱图等。
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), normalize_to_u8(arr))


def save_bgr(path: Path, img: np.ndarray) -> None:
    # 保存普通 BGR 彩色图。OpenCV 写图时默认就是 BGR 通道顺序。
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.clip(img, 0, 255).astype(np.uint8))


def block_view(ca: np.ndarray, block_shape: tuple[int, int] = BLOCK_SHAPE) -> tuple[np.ndarray, tuple[int, int]]:
    # 将 DWT 低频分量 ca 切成 4x4 小块。
    # 返回形状为 (块数量, 4, 4)，方便逐块做 DCT 和 SVD。
    block_h, block_w = block_shape
    rows = ca.shape[0] // block_h
    cols = ca.shape[1] // block_w
    cropped = ca[: rows * block_h, : cols * block_w]
    blocks = (
        cropped.reshape(rows, block_h, cols, block_w)
        .swapaxes(1, 2)
        .reshape(rows * cols, block_h, block_w)
    )
    return blocks, (rows, cols)


def blocks_to_ca(blocks: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    # block_view 的逆操作：把处理后的 4x4 小块重新拼回低频 ca 平面。
    rows, cols = grid_shape
    block_h, block_w = blocks.shape[1:]
    return blocks.reshape(rows, cols, block_h, block_w).swapaxes(1, 2).reshape(rows * block_h, cols * block_w)


def make_block_grid(blocks: np.ndarray, cols: int = 8, gap: int = 1, limit: int = 32) -> np.ndarray:
    # 把前若干个 4x4 低频块拼成网格图，用于报告中展示“分块”这一步。
    selected = blocks[: min(limit, len(blocks))]
    if selected.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    selected_u8 = np.array([normalize_to_u8(block) for block in selected])
    block_h, block_w = selected_u8.shape[1:]
    rows = int(np.ceil(len(selected_u8) / cols))
    canvas = np.full((rows * block_h + (rows - 1) * gap, cols * block_w + (cols - 1) * gap), 255, dtype=np.uint8)
    for idx, block in enumerate(selected_u8):
        r, c = divmod(idx, cols)
        y = r * (block_h + gap)
        x = c * (block_w + gap)
        canvas[y : y + block_h, x : x + block_w] = block
    return canvas


def save_dwt_overview(path: Path, ca: np.ndarray, hvd: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    # 将 DWT 的四个部分拼成一张图：
    # 左上 ca 为低频近似分量，右上/左下/右下分别为水平、垂直、对角高频细节。
    ch, cv, cd = hvd
    top = np.hstack([normalize_to_u8(ca), normalize_to_u8(ch)])
    bottom = np.hstack([normalize_to_u8(cv), normalize_to_u8(cd)])
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack([top, bottom]))


def modulate_singular_values(s: np.ndarray, bit: int, d1: float, d2: float) -> np.ndarray:
    # 核心水印嵌入公式：通过调整奇异值所在的模区间表示 0 或 1。
    # bit=0 时落在周期的 1/4 位置，bit=1 时落在周期的 3/4 位置。
    values = s.copy()
    values[0] = (values[0] // d1 + 1 / 4 + 1 / 2 * bit) * d1
    if d2 and len(values) > 1:
        values[1] = (values[1] // d2 + 1 / 4 + 1 / 2 * bit) * d2
    return values


class SimpleWatermarkDemo:
    def __init__(self, d1: float = 36, d2: float = 20, block_shape: tuple[int, int] = BLOCK_SHAPE):
        # d1/d2 是嵌入强度：越大越鲁棒，但对图像的改动也越明显。
        self.d1 = d1
        self.d2 = d2
        self.block_shape = block_shape

    def embed(self, img_bgr: np.ndarray, wm_bits: np.ndarray, trace_dir: Path | None = None) -> np.ndarray:
        # 嵌入主流程。输入是 BGR 图像和水印 bit 序列，输出是含水印 BGR 图像。
        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            raise ValueError("Expected a BGR image with shape (height, width, 3).")
        if wm_bits.size == 0:
            raise ValueError("Watermark bit array must not be empty.")

        img_float = img_bgr.astype(np.float32)
        # 1. BGR -> YUV：把亮度信息和色度信息分开，便于做频域处理。
        yuv = cv2.cvtColor(img_float, cv2.COLOR_BGR2YUV)
        if trace_dir:
            # 导出颜色空间转换后的可视化材料。
            save_bgr(trace_dir / "01_original_bgr.png", img_bgr)
            save_bgr(trace_dir / "02_yuv_as_bgr_preview.png", cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR))
            save_gray(trace_dir / "03_y_channel.png", yuv[:, :, 0])
            save_gray(trace_dir / "04_u_channel.png", yuv[:, :, 1])
            save_gray(trace_dir / "05_v_channel.png", yuv[:, :, 2])

        embedded_channels = []
        rows_and_cols = None
        sv_rows = []
        for channel in range(3):
            # 2. 对 Y/U/V 每个通道分别做 Haar DWT。
            # ca 是低频近似分量，hvd 是三个高频细节分量。
            ca, hvd = dwt2(yuv[:, :, channel], "haar")
            # 3. 只对低频 ca 分块并嵌入水印，高频 hvd 原样保留用于逆变换。
            blocks, grid_shape = block_view(ca, self.block_shape)
            if wm_bits.size >= len(blocks):
                raise ValueError(f"Watermark has {wm_bits.size} bits, but only {len(blocks)} blocks are available.")

            if channel == 0 and trace_dir:
                # 为了避免输出过多，这里只导出 Y 通道的 DWT/分块/DCT 演示图。
                save_gray(trace_dir / "06_dwt_ca_low_frequency_y.png", ca)
                save_gray(trace_dir / "07_dwt_ch_horizontal_y.png", hvd[0])
                save_gray(trace_dir / "08_dwt_cv_vertical_y.png", hvd[1])
                save_gray(trace_dir / "09_dwt_cd_diagonal_y.png", hvd[2])
                save_dwt_overview(trace_dir / "10_dwt_overview_y.png", ca, hvd)
                save_gray(trace_dir / "11_ca_first_blocks_grid.png", make_block_grid(blocks))
                save_gray(trace_dir / "12_first_ca_block.png", blocks[0])

            rows_and_cols = grid_shape
            new_blocks = blocks.copy()
            for idx, block in enumerate(blocks):
                # 水印按块循环嵌入：如果块数多于水印长度，同一个 bit 会重复出现。
                bit = int(wm_bits[idx % wm_bits.size])
                # 4. 对每个 4x4 低频块做 DCT，得到局部频域系数。
                block_dct = cv2.dct(block.astype(np.float32))
                # 5. 对 DCT 系数矩阵做 SVD，水印写入奇异值 s。
                u, s, vh = np.linalg.svd(block_dct)
                s_new = modulate_singular_values(s, bit, self.d1, self.d2)
                # 6. 用修改后的奇异值重构 DCT 系数，再逆 DCT 回到低频块。
                new_dct = u @ np.diag(s_new) @ vh
                new_blocks[idx] = cv2.idct(new_dct.astype(np.float32))

                if channel == 0 and idx < 8:
                    # 记录前几个块的奇异值变化，便于报告说明“bit 是如何写入 s[0]/s[1] 的”。
                    sv_rows.append([idx, bit, float(s[0]), float(s_new[0]), float(s[1]), float(s_new[1])])
                if channel == 0 and idx == 0 and trace_dir:
                    # DCT 系数范围差异很大，使用 log1p(abs(.)) 让频域图更容易看清。
                    save_gray(trace_dir / "13_first_block_dct_log.png", np.log1p(np.abs(block_dct)))
                    save_gray(trace_dir / "14_first_block_dct_after_svd_log.png", np.log1p(np.abs(new_dct)))
                    save_gray(trace_dir / "15_first_block_after_idct.png", new_blocks[idx])

            ca_embedded = ca.copy()
            # 7. 把所有修改后的 4x4 块拼回 ca 低频分量。
            ca_part = blocks_to_ca(new_blocks, grid_shape)
            ca_embedded[: ca_part.shape[0], : ca_part.shape[1]] = ca_part
            if channel == 0 and trace_dir:
                save_gray(trace_dir / "16_ca_after_embedding_y.png", ca_embedded)

            # 8. 修改后的 ca + 原高频 hvd 做逆 DWT，得到嵌入水印后的单通道图像。
            embedded_channel = idwt2((ca_embedded, hvd), "haar")
            embedded_channels.append(embedded_channel[: img_bgr.shape[0], : img_bgr.shape[1]])

        # 9. 三个 YUV 通道合并并转回 BGR，得到最终可保存/显示的含水印图。
        embedded_yuv = np.stack(embedded_channels, axis=2)
        embedded_bgr = cv2.cvtColor(embedded_yuv, cv2.COLOR_YUV2BGR)
        embedded_bgr = np.clip(embedded_bgr, 0, 255).astype(np.uint8)

        if trace_dir:
            # 保存最终结果、差异图和本次嵌入参数摘要。
            save_bgr(trace_dir / "17_watermarked_bgr.png", embedded_bgr)
            diff = np.abs(embedded_bgr.astype(np.int16) - img_bgr.astype(np.int16)).max(axis=2)
            save_gray(trace_dir / "18_difference_heatmap.png", diff)
            with (trace_dir / "19_first_blocks_singular_values.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["block_index", "wm_bit", "s0_before", "s0_after", "s1_before", "s1_after"])
                writer.writerows(sv_rows)
            with (trace_dir / "20_summary.txt").open("w", encoding="utf-8") as f:
                f.write(f"watermark_bits={wm_bits.size}\n")
                f.write(f"block_grid={rows_and_cols}\n")
                f.write(f"d1={self.d1}\n")
                f.write(f"d2={self.d2}\n")

        return embedded_bgr

    def extract_bits(self, embedded_bgr: np.ndarray, wm_size: int) -> np.ndarray:
        # 提取流程不需要原图：对含水印图做同样的 YUV、DWT、分块、DCT、SVD，
        # 再根据奇异值所在区间判断每个块里嵌入的是 0 还是 1。
        yuv = cv2.cvtColor(embedded_bgr.astype(np.float32), cv2.COLOR_BGR2YUV)
        block_votes = []
        for channel in range(3):
            ca, _ = dwt2(yuv[:, :, channel], "haar")
            blocks, _ = block_view(ca, self.block_shape)
            votes = np.zeros(len(blocks), dtype=np.float32)
            for idx, block in enumerate(blocks):
                block_dct = cv2.dct(block.astype(np.float32))
                _, s, _ = np.linalg.svd(block_dct)
                # 和嵌入公式对应：落在 d1 周期后半区判为 1，否则判为 0。
                bit0 = float(s[0] % self.d1 > self.d1 / 2)
                if self.d2 and len(s) > 1:
                    # s[1] 作为辅助判断，s[0] 权重更高。
                    bit1 = float(s[1] % self.d2 > self.d2 / 2)
                    votes[idx] = (3 * bit0 + bit1) / 4
                else:
                    votes[idx] = bit0
            block_votes.append(votes)

        vote_matrix = np.vstack(block_votes)
        extracted = np.zeros(wm_size, dtype=np.uint8)
        for bit_idx in range(wm_size):
            # 同一个水印 bit 会在多个块和三个通道里重复出现，这里求平均做多数判决。
            extracted[bit_idx] = int(vote_matrix[:, bit_idx::wm_size].mean() >= 0.5)
        return extracted


def parse_args() -> argparse.Namespace:
    # 命令行参数用于课堂演示：指定输入图、水印文本、输出目录和嵌入强度。
    parser = argparse.ArgumentParser(description="No-password DWT-DCT-SVD watermark demo with intermediate images.")
    parser.add_argument("--input", type=Path, default=Path("examples/pic/ori_img.jpeg"))
    parser.add_argument("--output-dir", type=Path, default=Path("course_project/output"))
    parser.add_argument("--watermark", default="SignalSystem")
    parser.add_argument("--d1", type=float, default=36)
    parser.add_argument("--d2", type=float, default=20)
    return parser.parse_args()


def main() -> None:
    # 串起完整 demo：读图 -> 文本转 bit -> 嵌入 -> 提取 -> 保存过程图片和结果。
    args = parse_args()
    img = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.input}")

    output_dir = args.output_dir
    trace_dir = output_dir / "trace"
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

    wm_bits = text_to_bits(args.watermark)
    demo = SimpleWatermarkDemo(d1=args.d1, d2=args.d2)
    embedded = demo.embed(img, wm_bits, trace_dir=trace_dir)
    extracted_bits = demo.extract_bits(embedded, wm_size=wm_bits.size)
    extracted_text = bits_to_text(extracted_bits)

    save_bgr(output_dir / "watermarked.png", embedded)
    with (output_dir / "result.txt").open("w", encoding="utf-8") as f:
        f.write(f"watermark_text={args.watermark}\n")
        f.write(f"watermark_bits={wm_bits.size}\n")
        f.write(f"extracted_text={extracted_text}\n")
        f.write(f"bit_errors={int(np.count_nonzero(wm_bits != extracted_bits))}\n")

    print(f"Watermark text: {args.watermark}")
    print(f"Extracted text: {extracted_text}")
    print(f"Bit errors: {int(np.count_nonzero(wm_bits != extracted_bits))}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
