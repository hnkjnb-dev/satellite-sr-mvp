from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


# 경로
lr_path = "outputs/baseline/sample_lr.png"
sr_path = "outputs/baseline/sample_sr.png"
hr_path = "outputs/baseline/sample_hr.png"

# 이미지 로드
lr_img = Image.open(lr_path).convert("RGB")
sr_img = Image.open(sr_path).convert("RGB")
hr_img = Image.open(hr_path).convert("RGB")

# Bicubic 업샘플링
bicubic_img = lr_img.resize(
    hr_img.size,
    Image.BICUBIC
)

# Figure 생성
plt.figure(figsize=(16, 4))

# LR
plt.subplot(1, 4, 1)
plt.imshow(lr_img)
plt.title(f"LR\n{lr_img.size}")
plt.axis("off")

# Bicubic
plt.subplot(1, 4, 2)
plt.imshow(bicubic_img)
plt.title("Bicubic")
plt.axis("off")

# SR
plt.subplot(1, 4, 3)
plt.imshow(sr_img)
plt.title("TinySR")
plt.axis("off")

# HR
plt.subplot(1, 4, 4)
plt.imshow(hr_img)
plt.title("HR (Ground Truth)")
plt.axis("off")

plt.tight_layout()

# 저장
save_path = "outputs/compare/comparison.png"
plt.savefig(save_path)

print(f"비교 이미지 저장 완료: {save_path}")