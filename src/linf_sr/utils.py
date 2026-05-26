# src/linf_sr/utils.py
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics = structural_similarity as ssim_func

def calculate_metrics(sr_tensor, hr_tensor):
    """
    초해상도 결과(SR)와 정답 이미지(HR)를 비교하여 PSNR 및 SSIM을 계산합니다.
    입력: PyTorch Tensor (Shape: [C, H, W], Range: 0~1)
    """
    # 1. 계산을 위해 텐서를 CPU 기반의 Numpy 배열로 변환 (HWC 구조)
    sr_img = sr_tensor.detach().cpu().permute(1, 2, 0).numpy()
    hr_img = hr_tensor.detach().cpu().permute(1, 2, 0).numpy()
    
    # 데이터 범위가 0~1 스케일이므로 data_range를 1.0으로 지정
    # 만약 채널 수가 다르면 channel_axis를 지정해야 합니다 (RGB는 channel_axis=2)
    psnr_score = psnr_func(hr_img, sr_img, data_range=1.0)
    ssim_score = ssim_func(hr_img, sr_img, data_range=1.0, channel_axis=2)
    
    return {"psnr": psnr_score, "ssim": ssim_score}