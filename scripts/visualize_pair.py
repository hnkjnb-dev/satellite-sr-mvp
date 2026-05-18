from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


hr_dir = Path("data/raw/hr")
lr_dir = Path("data/raw/lr")

# 첫 번째 파일 선택
hr_file = sorted(hr_dir.iterdir())[0]
lr_file = lr_dir / hr_file.name

print(f"HR 파일: {hr_file.name}")
print(f"LR 파일: {lr_file.name}")

# 이미지 로드
hr_img = Image.open(hr_file)
lr_img = Image.open(lr_file)

# 시각화
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(lr_img)
plt.title(f"LR: {lr_img.size}")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(hr_img)
plt.title(f"HR: {hr_img.size}")
plt.axis("off")

plt.tight_layout()
plt.show()