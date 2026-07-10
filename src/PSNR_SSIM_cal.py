import os
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 设置路径
original_folder = '/home/bml/storage/mnt/v-7c231cc5f5054b0a/org/data/DL3DV/1K_gt/train_test'        # 原始清晰图像文件夹
deblurred_folder = '/home/bml/storage/mnt/v-7c231cc5f5054b0a/org/code/Difix3D/Difix3D/outputs/model_80001_train_test'  # 去模糊后图像文件夹

psnr_values = []
ssim_values = []

# 遍历文件夹并处理每张图像
for filename in os.listdir(original_folder):

    original_path = os.path.join(original_folder, filename)
    deblurred_path = os.path.join(deblurred_folder, filename)

     # 读取图像并 resize 成 512x512
    img_original = Image.open(original_path).convert('RGB')
    img_deblurred = Image.open(deblurred_path).convert('RGB').resize((512, 512))

    # 读取图像
    img_original_np = np.array(img_original)
    img_deblurred_np = np.array(img_deblurred)

    # 计算 PSNR 和 SSIM
    current_psnr = psnr(img_original_np, img_deblurred_np, data_range=255)

    # 根据图像维度计算 SSIM
    if img_original_np.ndim == 3 and img_original_np.shape[2] == 3:
        # 彩色图，指定 channel_axis=2
        current_ssim = ssim(img_original_np, img_deblurred_np, win_size=11, multichannel=True,
                            channel_axis=2, data_range=255)
    else:
        raise ValueError(f"Unsupported image dimensions: {img_original_np.shape}")

    psnr_values.append(current_psnr)
    ssim_values.append(current_ssim)

# 输出平均值
avg_psnr = np.mean(psnr_values)
avg_ssim = np.mean(ssim_values)

print(f"Average PSNR: {avg_psnr:.4f} dB")
print(f"Average SSIM: {avg_ssim:.4f}")
