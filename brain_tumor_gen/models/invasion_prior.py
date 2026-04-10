"""
基于 Fisher-KPP 反应扩散方程的肿瘤侵袭先验
模拟肿瘤细胞的扩散与增殖行为
"""
import torch
import torch.nn.functional as F
import numpy as np


# 拉普拉斯卷积核（用于计算空间扩散）
LAPLACIAN_KERNEL = torch.tensor([
    [0,  1, 0],
    [1, -4, 1],
    [0,  1, 0]
], dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def compute_invasion_prior(
    mask: np.ndarray,
    diffusion_coeff: float = 0.1,
    proliferation_rate: float = 0.3,
    steps: int = 50,
    dt: float = 0.1,
    device: str = "cpu"
) -> np.ndarray:
    """
    输入医生绘制的初始肿瘤mask，输出符合生物学规律的侵袭扩散图

    Args:
        mask: 二值mask (H, W)，1表示肿瘤区域
        diffusion_coeff: 扩散系数，控制侵袭范围
        proliferation_rate: 增殖速率，控制肿瘤密度
        steps: 模拟步数
        dt: 时间步长
        device: 计算设备

    Returns:
        invasion_map: 侵袭概率图 (H, W)，值域 [0, 1]
    """
    kernel = LAPLACIAN_KERNEL.to(device)
    u = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    u = torch.clamp(u, 0, 1)

    for _ in range(steps):
        laplacian = F.conv2d(u, kernel, padding=1)
        # Fisher-KPP: du/dt = D*∇²u + r*u*(1-u)
        du = diffusion_coeff * laplacian + proliferation_rate * u * (1 - u)
        u = torch.clamp(u + dt * du, 0, 1)

    return u.squeeze().cpu().numpy()


def get_invasion_params_by_grade(grade: int) -> dict:
    """
    根据 WHO 肿瘤分级返回对应的侵袭参数
    Grade 1-2: 低级别，扩散慢
    Grade 3-4: 高级别，侵袭性强
    """
    params = {
        1: {"diffusion_coeff": 0.05, "proliferation_rate": 0.1, "steps": 30},
        2: {"diffusion_coeff": 0.08, "proliferation_rate": 0.2, "steps": 40},
        3: {"diffusion_coeff": 0.15, "proliferation_rate": 0.4, "steps": 60},
        4: {"diffusion_coeff": 0.25, "proliferation_rate": 0.6, "steps": 80},
    }
    return params.get(grade, params[2])
