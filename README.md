# hhDifix：双视图二阶段去雨精修

hhDifix 使用原始雨图 `R` 和冻结的初步去雨模型离线生成图 `P`，通过双视图
SD-Turbo/Difix 联合 attention 输出最终干净图：

```text
[view0: preliminary background P, view1: original rainy R] + prompt
                         -> two-view DiFix -> clean GT
```

主任务固定使用已对齐的 RGB 512×512 图像。`P` 必须在 DiFix 训练前由冻结的
UNet 离线生成；生成过程只读取雨图 `R`，不会使用该子集的 GT。

## 安装

```bash
pip install -r requirements.txt
```

需要支持 BF16 的 CUDA GPU。当前两阶段动态 optimizer 仅支持单 GPU；首次运行会从 Hugging Face 下载
`stabilityai/sd-turbo`。官方 3D/gsplat/nerfstudio 示例保留在 `examples/`，不属于
去雨入口，也不在主依赖中。

## 数据

### Rain13K 的 UNet/DiFix 分组划分

`src/split_rain13k.py` 按清晰目标图像文件的 SHA-256 分组，保证同一背景的不同雨纹版本不会跨越
UNet、DiFix 和验证子集。默认使用固定种子 3407 和 60%/30%/10% 比例，只生成清单，不移动或
复制原图：

```bash
python src/split_rain13k.py \
  --dataset_root /data/Rain13K/train/Rain13K \
  --output_dir /data/Rain13K/splits
```

输出包括 `unet_train.txt`、`difix_train.txt`、`validation.txt`、逐样本
`split_manifest.csv` 和含实际数量、背景组数量及泄漏检查结果的 `split_summary.json`。重新生成时必须
显式添加 `--overwrite`。

服务器端统一尺寸使用 `output/jupyter-notebook/prepare_rain13k_512.ipynb`。该 notebook 沿用
`degradation_test.ipynb` 的中心正方形裁剪和 LANCZOS 缩放方法，但会对 rainy/clean 配对图像使用
同一裁剪框，输出到新的 `Rain13K_512` 目录，并可按划分清单创建零拷贝硬链接训练视图。

当前服务器先只对 DiFix 训练子集建立独立视图。根据固定划分，
`difix_train.txt` 包含 4,113 张图：

```bash
cd /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/diff/hhdifix

python src/materialize_rain13k_views.py \
  --processed_root /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/data/Rain13K/train/Rain13K_512 \
  --split_dir /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/data/Rain13K/splits \
  --output_root /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/data/Rain13K/views_512 \
  --splits difix_train
```

默认使用硬链接，不重复占用图像空间；跨文件系统时添加 `--mode copy`。随后使用已经保存的
DCNv4 UNet 权重为这 4,113 张雨图生成同名 background PNG。专用脚本直接读取平铺的
`difix_train/input`，不调用 UNet 项目的旧 Dataset，也不会读取 `target`：

```bash
python src/generate_difix_background.py \
  --unet_root /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/DCNv4/general_decomp \
  --checkpoint /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/DCNv4/general_decomp/checkpoints/rain13k_unet_dcnv4_bg0722_retrain/best.pth \
  --input_dir /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/data/Rain13K/views_512/difix_train/input \
  --output_dir /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/data/Rain13K/views_512/difix_train/background \
  --expected_count 4113 \
  --batch_size 4 \
  --num_workers 4 \
  --device cuda
```

脚本默认使用 FP32，与原 UNet 推理路径一致。显存不足时先将 `--batch_size` 降为 2 或 1。
任务中断后用完全相同的命令加 `--resume`；脚本会验证并跳过已完成的同名 RGB PNG。
成功结束时会再次检查 `input` 与 `background` 是否都是完全相同的 4,113 个 stem。

验证集目录已经存在，可在开始 DiFix 验证前用同一脚本生成 background：

```bash
python src/generate_difix_background.py \
  --unet_root /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/DCNv4/general_decomp \
  --checkpoint /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/DCNv4/general_decomp/checkpoints/rain13k_unet_dcnv4_bg0722_retrain/best.pth \
  --input_dir /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/data/Rain13K/views_512/validation/input \
  --output_dir /home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/data/Rain13K/views_512/validation/background \
  --expected_count 1371 \
  --batch_size 4 \
  --num_workers 4 \
  --device cuda
```

三个目录必须具有完全相同的文件 stem；扩展名可以不同。每张图必须是 RGB
512×512。复制 `data/derain.example.json` 为 `data/derain.json` 并填写路径：

```json
{
  "train": {
    "image": "/data/train/rainy",
    "ref_image": "/data/train/preliminary",
    "target_image": "/data/train/gt",
    "prompt": "remove rain streaks and restore a clean natural image"
  },
  "test": {
    "image": "/data/test/rainy",
    "ref_image": "/data/test/preliminary",
    "target_image": "/data/test/gt",
    "prompt": "remove rain streaks and restore a clean natural image"
  }
}
```

Dataset 输出：

- `conditioning_pixel_values`: `[2,3,512,512]`，顺序 `[P,R]`，范围 `[-1,1]`
- `rainy_pixel_values`: `[3,512,512]`，仅用于报告原始雨图验证指标
- `target_pixel_values`: `[3,512,512]`，仅 GT，范围 `[-1,1]`
- `input_ids`: `[77]`

模型固定监督并输出 view 0，因此最终图像从 preliminary background 的 latent 开始精修；
原始退化雨图作为 view 1，通过双视图 attention 提供雨纹和内容线索。checkpoint 保存
`view_order=preliminary_rainy`，不能与 `[P,R-P]` checkpoint 混用或相互恢复训练。

默认增强为所有配对图同步水平翻转、20% reference dropout
（`[R,R] -> GT`）和 10% clean identity（`[GT,GT] -> GT`）。不进行 resize、
旋转、噪声合成或颜色抖动。

### 生成有符号雨残差视图

实验性 view 1 可以使用 `rainy - preliminary`。不要把负差值直接裁剪为黑色；生成脚本
将 `[-255,255]` 居中编码到 RGB PNG 的 `[0,255]`，因此 Dataset 归一化后近似得到
`(rainy-preliminary)/255`，零残差显示为中性灰色。

```bash
DATA=/home/bml/storage/mnt/v-zz4uoucip21b66el/PRP/Unet4Degradation/data/images5000_2

python src/build_rain_residual.py \
  --rainy_dir "$DATA/rain" \
  --preliminary_dir "$DATA/images5000_2_background" \
  --output_dir "$DATA/rain_minus_background" \
  --expected_count 4900

python src/build_rain_residual.py \
  --rainy_dir "$DATA/valid/rain" \
  --preliminary_dir "$DATA/valid/background" \
  --output_dir "$DATA/valid/rain_minus_background" \
  --expected_count 100
```

脚本严格按 stem 配对，检查 RGB、512×512 和预期数量，并始终输出无损 PNG。输出目录
已有同名 PNG 时默认拒绝覆盖；确认需要重建时显式添加 `--overwrite`。

## 训练

单卡：

```bash
bash src/train_singlegpu.sh
```

或直接执行：

```bash
accelerate launch --mixed_precision=bf16 src/train_difix.py \
  --dataset_path /data/Rain13K/derain.json \
  --output_dir outputs/hhdifix-rainy \
  --train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --checkpointing_steps 500 \
  --checkpoints_total_limit 5 \
  --eval_freq 500
```

默认有效 batch size 为 8。训练分为：

- 阶段 A：15,000 steps，仅 UNet LoRA rank 16，LR `5e-5`，warmup 500
- 阶段 B：5,000 steps，UNet LR `1e-5`，启用 VAE decoder LoRA rank 4 和四层 encoder→decoder skip 1×1 卷积，LR `1e-5`

固定 `timestep=199`、`CFG=1.0`、prompt dropout 关闭、gradient clip 1.0。
损失为：

```text
Charbonnier + 0.2 SSIM + 0.1 LPIPS + 0.05 Sobel
```

阶段 A 结束时先保存 `stage-a-final.pt`，然后在原 AdamW 上降低 UNet LR 并通过
`add_param_group()` 加入 VAE decoder LoRA 和 skip 卷积，因此 UNet Adam 动量会保留。skip 卷积以零权重初始化，
在阶段 A 不改变原始 VAE 输出，阶段 B 才参与训练。checkpoint 使用 v3 schema：
`checkpoints/resume/checkpoint-*.pt` 保存 LoRA、skip 权重、optimizer、scheduler、
阶段和 global step，自动只保留最近 5 个；`best.pt` 和 `final.pt` 是不含 optimizer 的
轻量推理权重。所有 checkpoint 都不保存完整 SD-Turbo 权重。由于 v2 checkpoint 不含 skip 权重，不能直接用于 v3 恢复训练或推理。

### 续训

可以传入具体 checkpoint，也可以传入 checkpoint 目录（自动选择最大步数）：

```bash
accelerate launch --mixed_precision=bf16 src/train_difix.py \
  --dataset_path /data/Rain13K/derain.json \
  --output_dir outputs/hhdifix-rainy \
  --resume outputs/hhdifix-rainy/checkpoints
```

目录恢复会从 `checkpoints/resume/` 选择数字最大的 checkpoint。也可以显式从
`stage-a-final.pt` 恢复，加载阶段 A optimizer 后会自动完成 A→B 转换。数据、阶段
步数、LoRA rank、prompt 和梯度累积参数应与原训练保持一致。

## 推理

```bash
python src/inference_difix.py \
  --rainy_dir /data/test/rainy \
  --preliminary_dir /data/test/preliminary \
  --checkpoint outputs/hhdifix-rainy/checkpoints/final.pt \
  --output_dir outputs/hhdifix-rainy/results
```

推理严格按 stem 配对并检查同尺寸，使用 VAE posterior `mode()`，因此固定权重和
输入下输出确定。默认 prompt 来自 checkpoint；只有显式传入 `--prompt` 时才覆盖。

## 评测

```bash
python src/evaluate_img.py \
  --rainy_dir /data/test/rainy \
  --preliminary_dir /data/test/preliminary \
  --final_dir outputs/hhdifix-rainy/results \
  --gt_dir /data/test/gt \
  --output outputs/hhdifix-rainy/metrics.json
```

训练期 validation 和独立评测都会报告 Rainy、Preliminary、Final 各自的
PSNR/SSIM/LPIPS，以及 Final 相对 Preliminary 的 ΔPSNR、ΔLPIPS 和逐样本胜率。
`best.pt` 首先按最高 Final PSNR 选择，相同时依次比较更低 LPIPS 和更高 SSIM。

## 测试

```bash
pytest -q
python -m compileall -q src tests
```

完整验收还需在目标 GPU 上执行：小样本过拟合、双视图 forward、LoRA 梯度、
checkpoint round-trip、resume 学习率连续性以及端到端训练/推理/评测。

## License

代码沿用原项目 [LICENSE.txt](LICENSE.txt)，基础模型另受 SD-Turbo 许可证约束。
