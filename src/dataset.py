import json
import torch
from PIL import Image
import torchvision.transforms.functional as F
import os
from torchvision import transforms
import random

def add_combined_noise_torch(
    image,
    pseudo_mask,
    stddev=0.1,
    scale=0.3,
    sparsity=0.5,
    gan=0.4,
    poisson_weight=0.4,
    poisson_L=30,
    brightness_sensitive=True
):
    """
    对 torch.Tensor 图像添加组合噪声（高斯 + 泊松 + 稀疏 + 亮度调节）

    Args:
        image (Tensor): [C,H,W] 或 [B,C,H,W]，值范围 [0,1]
        stddev (float): 高斯噪声标准差
        scale (float): 控制块状程度
        sparsity (float): 0~1，控制稀疏性
        gan (float): 高斯噪声强度
        poisson_weight (float): 泊松噪声强度
        poisson_L (float): 泊松噪声放大系数（越小噪声越大）
        brightness_sensitive (bool): 是否基于亮度调整噪声强度
    Returns:
        Tensor: 噪声后的图像，值范围 [0,1]
    """

    is_batched = image.dim() == 4
    if not is_batched:
        image = image.unsqueeze(0)

    B, C, H, W = image.shape
    h_small, w_small = max(1, int(H * scale)), max(1, int(W * scale))

    # === 亮度感知掩码 ===
    if brightness_sensitive:
        with torch.no_grad():
            gray = image.mean(dim=1, keepdim=True)  # [B,1,H,W]
            brightness_factor = 1.0 - gray  # 暗处值大，亮处值小
            brightness_factor = torch.nn.functional.interpolate(brightness_factor, size=(h_small, w_small), mode='bilinear', align_corners=False)
    else:
        brightness_factor = torch.ones((B, 1, h_small, w_small), device=image.device)

    # === 高斯噪声 ===
    noise = torch.randn((B, C, h_small, w_small), device=image.device) * stddev
    if sparsity is not None:
        mask = (torch.rand((B, 1, h_small, w_small), device=image.device) < sparsity).float()
        noise *= mask
    noise *= brightness_factor
    noise = torch.nn.functional.interpolate(noise, size=(H, W), mode='bilinear', align_corners=False)
    gauss_noisy = gan * noise

    # === 泊松噪声 ===
    poisson_input = torch.clamp(image * poisson_L, min=0)
    poisson_noise = (torch.poisson(poisson_input) / poisson_L) - image
    if brightness_sensitive:
        poisson_noise *= torch.nn.functional.interpolate(brightness_factor, size=(H, W), mode='bilinear', align_corners=False)

    

    poisson_noisy = poisson_weight * poisson_noise
    total_noise = gauss_noisy + poisson_noisy

    if pseudo_mask is not None:
        pseudo_mask = pseudo_mask.view(B, 1, 1, 1).float()  # [B,1,1,1]
        total_noise = total_noise * pseudo_mask

    # === 合成并裁剪 ===
    noisy_image = image + total_noise
    noisy_image = torch.clamp(noisy_image, 0.0, 1.0)

    return noisy_image if is_batched else noisy_image.squeeze(0)

class RandomSubsetColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p_each=0.5):
        self.transforms = [
            transforms.ColorJitter(brightness=brightness),
            transforms.ColorJitter(contrast=contrast),
            transforms.ColorJitter(saturation=saturation),
            transforms.ColorJitter(hue=hue)
        ]
        self.p_each = p_each

    def __call__(self, img):
        # 随机挑选部分变换
        selected = [t for t in self.transforms if random.random() < self.p_each]
        # 随机打乱顺序
        random.shuffle(selected)
        # 应用
        for t in selected:
            img = t(img)
        return img

class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, height=512, width=512, tokenizer=None):
        super().__init__()

        # 读取 JSON 文件
        with open(dataset_path, 'r') as f:
            json_data = json.load(f)[split]

        self.image_dir = json_data['image']
        self.target_dir = json_data['target_image']
        self.ref_dir = json_data.get('ref_image', None)
        self.prompt = json_data['prompt']

        # 获取所有图像文件名（按顺序排列）
        self.image_files = sorted([
            os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.target_files = sorted([
            os.path.join(self.target_dir, f) for f in os.listdir(self.target_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])

        if self.ref_dir:
            self.ref_files = sorted([
                os.path.join(self.ref_dir, f) for f in os.listdir(self.ref_dir)
                if f.endswith(('.png', '.jpg', '.jpeg'))
            ])
        else:
            self.ref_files = [None] * len(self.image_files)
        
        self.ref_dir2 = json_data.get('ref_image2', None)
        if self.ref_dir2 is not None:
            self.image_dir2 = json_data['image2']
            self.target_dir2 = json_data['target_image2']

            # 获取所有图像文件名（按顺序排列）
            self.image_files2 = sorted([
                os.path.join(self.image_dir2, f) for f in os.listdir(self.image_dir2)
                if f.endswith(('.png', '.jpg', '.jpeg'))
            ])
            self.target_files2 = sorted([
                os.path.join(self.target_dir2, f) for f in os.listdir(self.target_dir2)
                if f.endswith(('.png', '.jpg', '.jpeg'))
            ])

            if self.ref_dir2:
                self.ref_files2 = sorted([
                    os.path.join(self.ref_dir2, f) for f in os.listdir(self.ref_dir2)
                    if f.endswith(('.png', '.jpg', '.jpeg'))
                ])
            print(len(self.image_files2))
            print(len(self.ref_files2))
            print(len(self.target_files2))
            # self.image_files += self.image_files2 * 100
            # self.ref_files += self.ref_files2 * 100
            # self.target_files += self.target_files2 * 100
            self.image_files = self.image_files * 110 + self.image_files2
            self.ref_files = self.ref_files * 110 + self.ref_files2
            self.target_files = self.target_files * 110 + self.target_files2


        assert len(self.image_files) == len(self.target_files) == len(self.ref_files), \
            "输入、目标、参考图像数量必须一致"

        self.image_size = (height, width)
        self.image_size_small = (128, 128)
        self.tokenizer = tokenizer
        self.color_trans = RandomSubsetColorJitter()
        

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        input_path = self.image_files[idx]
        target_path = self.target_files[idx]
        ref_path = self.ref_files[idx]

        if self.ref_dir2 is not None and target_path in self.target_files2:
            pseudo=True
        else:
            pseudo=False

        # print(input_path)
        # print(target_path)
        # print(ref_path)

        try:
            input_img = Image.open(input_path).convert("RGB")
            target_img = Image.open(target_path).convert("RGB")
        except Exception as e:
            print(f"Error loading images: {input_path}, {target_path}")
            return self.__getitem__((idx + 1) % len(self))

        # 图像预处理
        input_tensor = F.to_tensor(input_img)

        # 原来的参数
        if pseudo and random.random() < 0.6:
            input_tensor = add_combined_noise_torch(
            input_tensor,
            None,
            stddev=random.uniform(0.1, 0.4),
            gan=random.uniform(0.1, 0.5),
            poisson_weight=random.uniform(0.1, 0.4),
            sparsity=random.uniform(0.2, 0.5),
            scale=random.uniform(0.2, 0.5),
            brightness_sensitive=True
            )

        # # Phlatcam 的参数
        # if pseudo and random.random() < 0.6:
        #     input_tensor = add_combined_noise_torch(
        #     input_tensor,
        #     None,
        #     stddev=0.017,#
        #     gan=random.uniform(0.5, 0.6),
        #     poisson_weight=random.uniform(0.01, 0.02),
        #     poisson_L=random.uniform(20, 40),
        #     sparsity=random.uniform(0.8, 0.9),
        #     scale=random.uniform(0.1, 0.2),
        #     brightness_sensitive=False
        #     )

        input_tensor = F.resize(input_tensor, self.image_size_small)
        input_tensor = F.resize(input_tensor, self.image_size)
        

        target_tensor = F.to_tensor(target_img)
        target_tensor = F.resize(target_tensor, self.image_size)
        target_tensor = F.normalize(target_tensor, mean=[0.5], std=[0.5])

        # 处理参考图像
        if ref_path is not None:
            ref_img = Image.open(ref_path).convert("RGB")
            ref_tensor = F.to_tensor(ref_img)
            ref_tensor = F.resize(ref_tensor, self.image_size)
            ref_tensor = F.normalize(ref_tensor, mean=[0.5], std=[0.5])

            
            target_tensor = torch.stack([target_tensor, ref_tensor], dim=0)
        else:
            input_tensor = input_tensor.unsqueeze(0)
            target_tensor = target_tensor.unsqueeze(0)

        if pseudo and random.random() < 0.1:
            input_tensor = self.color_trans(input_tensor)
        
        
        input_tensor = F.normalize(input_tensor, mean=[0.5], std=[0.5])
        input_tensor = torch.stack([input_tensor, ref_tensor], dim=0)
        out = {
            "output_pixel_values": target_tensor,
            "conditioning_pixel_values": input_tensor,
            "caption": self.prompt,
        }

        if self.tokenizer is not None:
            input_ids = self.tokenizer(
                self.prompt,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids
            out["input_ids"] = input_ids

        return out