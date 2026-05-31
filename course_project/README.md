# 大作业简化演示代码

本目录提供一条课程展示用的 DWT-DCT-SVD 水印流程。它和原始 `blind_watermark` 包分开，目的是突出核心算法步骤：

```text
BGR 图像 -> YUV -> DWT -> ca 低频分块 -> DCT -> SVD -> 修改奇异值 -> 逆变换
```

## 与原项目的区别

- 不使用 `password_img`。
- 不使用 `password_wm`。
- 不打乱 DCT 系数。
- 不打乱水印 bit。
- 只保留频域水印核心流程，方便课程报告和课堂演示。

## 运行方式

在仓库根目录执行：

```bash
python course_project/simple_watermark_demo.py --input examples/pic/ori_img.jpeg --watermark SignalSystem
```

输出目录默认为：

```text
course_project/output/
```

其中：

- `watermarked.png`：最终含水印图像。
- `result.txt`：原始水印、提取水印和 bit 错误数。
- `trace/`：算法过程图片和辅助表格。

## 中间过程图片

`trace/` 中会生成：

- `01_original_bgr.png`：原始 BGR 图像。
- `02_yuv_as_bgr_preview.png`：YUV 转换后的预览图。
- `03_y_channel.png`、`04_u_channel.png`、`05_v_channel.png`：Y/U/V 三个通道。
- `06_dwt_ca_low_frequency_y.png`：Y 通道 DWT 低频 `ca`。
- `07_dwt_ch_horizontal_y.png`：水平高频分量。
- `08_dwt_cv_vertical_y.png`：垂直高频分量。
- `09_dwt_cd_diagonal_y.png`：对角高频分量。
- `10_dwt_overview_y.png`：DWT 四个分量拼图。
- `11_ca_first_blocks_grid.png`：低频 `ca` 的前若干个 4x4 分块。
- `12_first_ca_block.png`：第一个低频块。
- `13_first_block_dct_log.png`：第一个低频块 DCT 后的频域图。
- `14_first_block_dct_after_svd_log.png`：修改奇异值后的 DCT 频域图。
- `15_first_block_after_idct.png`：逆 DCT 后的第一个低频块。
- `16_ca_after_embedding_y.png`：嵌入后的 Y 通道低频 `ca`。
- `17_watermarked_bgr.png`：最终含水印图。
- `18_difference_heatmap.png`：原图与含水印图差异热力图。
- `19_first_blocks_singular_values.csv`：前几个块修改前后的奇异值。
- `20_summary.txt`：本次实验参数摘要。

这些文件可以直接作为大作业报告中的算法过程展示素材。
