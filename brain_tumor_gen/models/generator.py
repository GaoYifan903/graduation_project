"""
生成模型封装
三种模式：
  - vae:  轻量 VAE (~330MB)，Demo结构图 + VAE纹理增强
  - ldm:  完整 LDM + ControlNet (~5GB)
  - demo: 纯numpy合成，无需任何模型
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import io, base64
from pathlib import Path

from .invasion_prior import compute_invasion_prior, get_invasion_params_by_grade
from .concept_bottleneck import ConceptBottleneck, concept_scores_to_report
from .texture_transfer import apply_texture_transfer


class TumorGenerator:
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        self.image_size = config.IMAGE_SIZE
        self.concept_net = ConceptBottleneck().to(self.device)
        self.vae = None
        self.mode = "demo"
        self._load_model()

    def _load_model(self):
        """优先尝试 VAE，失败则 demo 模式；同时检测 BraTS 数据"""
        # 检测 BraTS 数据
        brats_root = getattr(self.config, "BRATS_DATA_ROOT", None)
        self.brats_root = brats_root if brats_root and Path(brats_root).exists() else None
        if self.brats_root:
            print(f"[Generator] 检测到 BraTS 数据: {self.brats_root}，将使用真实纹理迁移")

        try:
            from diffusers import AutoencoderKL
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            vae_path = getattr(self.config, "VAE_PATH", "stabilityai/sd-vae-ft-mse")
            print(f"[Generator] 正在加载 VAE: {vae_path}")
            self.vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype).to(self.device)
            self.vae.eval()
            self.mode = "vae"
            print("[Generator] VAE 加载成功，使用 VAE 纹理增强模式")
        except Exception as e:
            print(f"[Generator] VAE 加载失败，进入 Demo 模式: {e}")
            self.mode = "demo"

    # ------------------------------------------------------------------ #
    #  主接口
    # ------------------------------------------------------------------ #
    def generate(self, mask, size_cm, shape, grade, edema_range, enhancement):
        # 1. 侵袭先验
        invasion_params = get_invasion_params_by_grade(grade)
        invasion_map = compute_invasion_prior(mask, **invasion_params, device="cpu")

        # 2. 先用 demo 合成结构图
        struct_img = self._generate_demo(
            mask, invasion_map, size_cm, shape, grade, edema_range, enhancement
        )

        # 3. 纹理增强（优先用真实BraTS纹理迁移，其次VAE，最后纯demo）
        if self.brats_root:
            modality = "t1ce" if enhancement else "t1"
            generated = apply_texture_transfer(
                struct_img, mask, self.brats_root,
                modality=modality, fft_alpha=0.65
            )
        elif self.mode == "vae":
            generated = self._vae_enhance(struct_img, noise_strength=0.35)
        else:
            generated = struct_img

        # 4. 概念分析
        z = torch.randn(1, 512).to(next(self.concept_net.parameters()).device)
        concept_scores = self.concept_net(z)

        # 5. 热图 & 报告
        heatmap = self._make_heatmap(mask, invasion_map, concept_scores)
        report = concept_scores_to_report(concept_scores, grade, shape)

        return {
            "image_b64": self._to_b64(generated),
            "heatmap_b64": self._to_b64(heatmap),
            "invasion_map_b64": self._to_b64(self._invasion_to_img(invasion_map)),
            "report": report
        }

    # ------------------------------------------------------------------ #
    #  VAE 纹理增强
    # ------------------------------------------------------------------ #
    def _vae_enhance(self, pil_img: Image.Image, noise_strength: float = 0.35) -> Image.Image:
        """
        将结构图编码进 VAE 隐空间，加少量噪声后解码
        让图像获得类似真实 MRI 的纹理，同时保留结构
        """
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        # 预处理：PIL -> tensor [-1, 1]
        img = pil_img.resize((self.image_size, self.image_size)).convert("RGB")
        x = torch.tensor(np.array(img), dtype=dtype).permute(2, 0, 1).unsqueeze(0)
        x = (x / 127.5 - 1.0).to(self.device)

        with torch.no_grad():
            # 编码
            posterior = self.vae.encode(x).latent_dist
            z = posterior.sample()

            # 在隐空间加噪，增加纹理多样性
            noise = torch.randn_like(z) * noise_strength
            z_noisy = z + noise

            # 解码
            decoded = self.vae.decode(z_noisy).sample

        # 后处理：tensor -> PIL
        decoded = decoded.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        decoded = np.clip((decoded + 1.0) * 127.5, 0, 255).astype(np.uint8)
        return Image.fromarray(decoded)

    # ------------------------------------------------------------------ #
    #  Demo 结构合成
    # ------------------------------------------------------------------ #
    def _generate_demo(self, mask, invasion_map, size_cm, shape, grade, edema_range, enhancement):
        H, W = self.image_size, self.image_size
        img = np.zeros((H, W), dtype=np.float32)

        # 脑组织背景（椭圆）
        cy, cx = H // 2, W // 2
        Y, X = np.ogrid[:H, :W]
        brain_mask = ((X - cx)**2 / (cx * 0.85)**2 + (Y - cy)**2 / (cy * 0.9)**2) <= 1
        img[brain_mask] = 0.35 + np.random.normal(0, 0.03, img.shape)[brain_mask]

        # 水肿区（T2高信号）
        img += invasion_map * edema_range * 0.15 * 0.4

        # 肿瘤核心
        tumor_region = mask > 0.5
        img[tumor_region] = 0.6 + np.random.normal(0, 0.05, img.shape)[tumor_region]

        # 强化环
        if enhancement:
            from scipy.ndimage import binary_dilation
            ring = binary_dilation(tumor_region, iterations=3) & ~tumor_region
            img[ring] = 0.85 + np.random.normal(0, 0.03, img.shape)[ring]

        # 坏死核心（高级别）
        if grade >= 3 and tumor_region.any():
            from scipy.ndimage import binary_erosion
            eroded = binary_erosion(tumor_region, iterations=max(1, int(size_cm)))
            img[eroded] = 0.1 + np.random.normal(0, 0.02, img.shape)[eroded]

        img = np.clip(img, 0, 1)
        return Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")

    # ------------------------------------------------------------------ #
    #  可视化
    # ------------------------------------------------------------------ #
    def _make_heatmap(self, mask, invasion_map, concept_scores):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="#1a1a2e")
        for ax, title, d, cmap in zip(
            axes,
            ["位置Mask", "侵袭先验", "概念权重"],
            [mask, invasion_map, np.array(list(concept_scores.values())).reshape(2, 3)],
            ["Blues", "hot", "YlOrRd"]
        ):
            ax.imshow(d, cmap=cmap, aspect="auto")
            ax.set_title(title, color="white", fontsize=10)
            ax.axis("off")

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor="#1a1a2e")
        plt.close()
        buf.seek(0)
        return Image.open(buf)

    def _invasion_to_img(self, invasion_map):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(invasion_map, cmap="hot")
        ax.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
        return Image.open(buf)

    def _to_b64(self, img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
