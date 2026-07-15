# hhDifix：双视图二阶段去雨精修

hhDifix 使用原始雨图 `R` 和冻结的初步去雨模型离线生成图 `P`，通过双视图
SD-Turbo/Difix 联合 attention 输出最终干净图：

```text
[preliminary P, rainy R] + prompt -> two-view Difix -> clean GT
```

主任务固定使用已对齐的 RGB 512×512 图像。项目不负责训练或调用初步去雨
UNet，也不会在训练过程中读取验证/测试 GT 来生成 `P`。

## 安装

```bash
pip install -r requirements.txt
```

需要支持 BF16 的 CUDA GPU。当前两阶段动态 optimizer 仅支持单 GPU；首次运行会从 Hugging Face 下载
`stabilityai/sd-turbo`。官方 3D/gsplat/nerfstudio 示例保留在 `examples/`，不属于
去雨入口，也不在主依赖中。

## 数据

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
- `target_pixel_values`: `[3,512,512]`，仅 GT，范围 `[-1,1]`
- `input_ids`: `[77]`

模型固定监督并输出 view 0，因此最终图像从 preliminary 的 latent 开始精修；原始雨图
作为 view 1，通过双视图 attention 提供退化线索。新 checkpoint 会保存
`view_order=preliminary_rainy`。没有该字段的旧 checkpoint 按旧顺序
`rainy_preliminary` 加载，不能用于恢复新顺序训练。

默认增强为三图同步水平翻转、20% reference dropout（`P=R`）和 10% clean
identity（`[GT,GT] -> GT`）。不进行 resize、旋转、噪声合成或颜色抖动。

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
  --dataset_path data/derain.json \
  --output_dir outputs/hhdifix \
  --train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --checkpointing_steps 500 \
  --checkpoints_total_limit 5 \
  --eval_freq 500
```

默认有效 batch size 为 8。训练分为：

- 阶段 A：15,000 steps，仅 UNet LoRA rank 16，LR `5e-5`，warmup 500
- 阶段 B：5,000 steps，UNet LR `1e-5`，启用 VAE decoder LoRA rank 4，LR `1e-5`

固定 `timestep=199`、`CFG=1.0`、prompt dropout 关闭、gradient clip 1.0。
损失为：

```text
Charbonnier + 0.2 SSIM + 0.1 LPIPS + 0.05 Sobel
```

阶段 A 结束时先保存 `stage-a-final.pt`，然后在原 AdamW 上降低 UNet LR 并通过
`add_param_group()` 加入 VAE decoder LoRA，因此 UNet Adam 动量会保留。checkpoint
使用 v2 schema：`checkpoints/resume/checkpoint-*.pt` 保存 LoRA、optimizer、scheduler、
阶段和 global step，自动只保留最近 5 个；`best.pt` 和 `final.pt` 是不含 optimizer 的
轻量推理权重。所有 checkpoint 都不保存完整 SD-Turbo 权重。

### 续训

可以传入具体 checkpoint，也可以传入 checkpoint 目录（自动选择最大步数）：

```bash
accelerate launch --mixed_precision=bf16 src/train_difix.py \
  --dataset_path data/derain.json \
  --output_dir outputs/hhdifix \
  --resume outputs/hhdifix/checkpoints
```

目录恢复会从 `checkpoints/resume/` 选择数字最大的 checkpoint。也可以显式从
`stage-a-final.pt` 恢复，加载阶段 A optimizer 后会自动完成 A→B 转换。数据、阶段
步数、LoRA rank、prompt 和梯度累积参数应与原训练保持一致。

## 推理

```bash
python src/inference_difix.py \
  --rainy_dir /data/test/rainy \
  --preliminary_dir /data/test/preliminary \
  --checkpoint outputs/hhdifix/checkpoints/final.pt \
  --output_dir outputs/hhdifix/results
```

推理严格按 stem 配对并检查同尺寸，使用 VAE posterior `mode()`，因此固定权重和
输入下输出确定。默认 prompt 来自 checkpoint；只有显式传入 `--prompt` 时才覆盖。

## 评测

```bash
python src/evaluate_img.py \
  --rainy_dir /data/test/rainy \
  --preliminary_dir /data/test/preliminary \
  --final_dir outputs/hhdifix/results \
  --gt_dir /data/test/gt \
  --output outputs/hhdifix/metrics.json
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
