"""
基于真实 BraTS MRI 的纹理迁移
方法：直方图匹配 + FFT 频域纹理混合 + 结构保护
无需任何深度学习模型
"""
import numpy as np
from PIL import Image
from pathlib import Path
import random


def load_brats_slice(data_root: str, modality: str = "t1ce") -> np.ndarray:
    """
    从 BraTS 数据集随机加载一张轴向切片
    返回归一化到 [0,1] 的 2D numpy 数组
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("请先安装 nibabel: pip install nibabel")

    root = Path(data_root)
    cases = [d for d in root.iterdir() if d.is_dir()]
    if not cases:
        raise FileNotFoundError(f"未找到病例目录: {data_root}")

    # 随机选一个病例
    case = random.choice(cases)
    nii_files = list(case.glob(f"*_00_*_{modality}.nii.gz"))
    if not nii_files:
        nii_files = list(case.glob(f"*_{modality}.nii.gz"))
    if not nii_files:
        raise FileNotFoundError(f"未找到 {modality} 文件: {case}")

    vol = nib.load(str(nii_files[0])).get_fdata()  # (H, W, D)

    # 取中间1/3的切片（肿瘤最常出现的区域）
    d = vol.shape[2]
    slices = vol[:, :, d//3: 2*d//3]

    # 找非空切片（脑组织占比 > 20%）
    for _ in range(20):
        idx = random.randint(0, slices.shape[2] - 1)
        sl = slices[:, :, idx]
        if (sl > 0).mean() > 0.2:
            break

    # 归一化
    sl = sl.astype(np.float32)
    p1, p99 = np.percentile(sl[sl > 0], [1, 99]) if (sl > 0).any() else (0, 1)
    sl = np.clip((sl - p1) / (p99 - p1 + 1e-8), 0, 1)
    return sl


def histogram_match(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    将 source 图像的灰度分布匹配到 reference
    保留 source 的结构，获得 reference 的灰度风格
    """
    src_flat = source.flatten()
    ref_flat = reference.flatten()

    # 只对非零区域做匹配
    src_mask = src_flat > 0.01
    ref_mask = ref_flat > 0.01

    if src_mask.sum() == 0 or ref_mask.sum() == 0:
        return source

    # 计算 CDF
    src_hist, bins = np.histogram(src_flat[src_mask], bins=256, range=(0, 1))
    ref_hist, _    = np.histogram(ref_flat[ref_mask], bins=256, range=(0, 1))

    src_cdf = np.cumsum(src_hist).astype(np.float64)
    ref_cdf = np.cumsum(ref_hist).astype(np.float64)
    src_cdf /= src_cdf[-1]
    ref_cdf /= ref_cdf[-1]

    # 建立映射表
    mapping = np.interp(src_cdf, ref_cdf, np.linspace(0, 1, 256))
    bin_idx = np.digitize(src_flat, bins[:-1]) - 1
    bin_idx = np.clip(bin_idx, 0, 255)

    result = source.copy()
    result_flat = result.flatten()
    result_flat[src_mask] = mapping[bin_idx[src_mask]]
    return result_flat.reshape(source.shape)


def fft_texture_blend(
    struct_img: np.ndarray,
    real_mri: np.ndarray,
    alpha: float = 0.7
) -> np.ndarray:
    """
    FFT 频域混合：
    - 低频（结构）来自 struct_img（保留肿瘤位置/形状）
    - 高频（纹理）来自 real_mri（真实MRI纹理）
    alpha: 高频混合比例，越大纹理越真实
    """
    H, W = struct_img.shape

    # resize real_mri 到相同尺寸
    real_resized = np.array(
        Image.fromarray((real_mri * 255).astype(np.uint8)).resize((W, H))
    ).astype(np.float32) / 255.0

    # FFT
    f_struct = np.fft.fft2(struct_img)
    f_real   = np.fft.fft2(real_resized)

    f_struct_shift = np.fft.fftshift(f_struct)
    f_real_shift   = np.fft.fftshift(f_real)

    # 低频掩码（中心区域）
    cy, cx = H // 2, W // 2
    radius = min(H, W) // 6  # 低频半径
    Y, X = np.ogrid[:H, :W]
    low_freq_mask = ((X - cx)**2 + (Y - cy)**2) <= radius**2
    high_freq_mask = ~low_freq_mask

    # 混合：低频保留结构，高频引入真实纹理
    f_blend = f_struct_shift.copy()
    f_blend[high_freq_mask] = (
        (1 - alpha) * f_struct_shift[high_freq_mask] +
        alpha       * f_real_shift[high_freq_mask]
    )

    # 逆 FFT
    f_blend_ishift = np.fft.ifftshift(f_blend)
    result = np.abs(np.fft.ifft2(f_blend_ishift))
    return np.clip(result, 0, 1)


def structure_protect_blend(
    textured: np.ndarray,
    struct_img: np.ndarray,
    mask: np.ndarray,
    protect_strength: float = 0.6
) -> np.ndarray:
    """
    在肿瘤边界区域加强结构保护，防止纹理迁移破坏肿瘤形态
    """
    from scipy.ndimage import binary_dilation
    # 肿瘤边界区域
    if mask.max() > 0:
        boundary = binary_dilation(mask > 0.5, iterations=5).astype(np.float32)
        boundary -= (mask > 0.5).astype(np.float32)
        boundary = np.clip(boundary, 0, 1)
        # 边界区域更多保留结构
        result = textured * (1 - boundary * protect_strength) + struct_img * (boundary * protect_strength)
    else:
        result = textured
    return np.clip(result, 0, 1)


def apply_texture_transfer(
    struct_pil: Image.Image,
    mask: np.ndarray,
    brats_data_root: str,
    modality: str = "t1ce",
    fft_alpha: float = 0.65
) -> Image.Image:
    """
    完整纹理迁移流程

    Args:
        struct_pil: Demo 合成的结构图 (PIL RGB)
        mask: 肿瘤位置 mask (H, W)
        brats_data_root: BraTS 数据根目录
        modality: 使用的模态 (t1/t1ce/t2/flair)
        fft_alpha: 高频纹理混合强度 [0,1]

    Returns:
        纹理迁移后的 PIL 图像
    """
    size = struct_pil.size[0]

    # 转灰度
    struct_gray = np.array(struct_pil.convert("L")).astype(np.float32) / 255.0

    # 加载真实 MRI 切片
    real_slice = load_brats_slice(brats_data_root, modality)

    # 1. 直方图匹配
    matched = histogram_match(struct_gray, real_slice)

    # 2. FFT 频域纹理混合
    textured = fft_texture_blend(matched, real_slice, alpha=fft_alpha)

    # 3. 结构保护
    result = structure_protect_blend(textured, matched, mask)

    # 转回 RGB PIL
    result_uint8 = (result * 255).astype(np.uint8)
    result_rgb = np.stack([result_uint8] * 3, axis=-1)
    return Image.fromarray(result_rgb).resize((size, size))
