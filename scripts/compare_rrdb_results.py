from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

output_dir = Path("outputs/rrdb_minimal")

lr_img = Image.open(output_dir / "sample_lr.png").convert("RGB")
hr_img = Image.open(output_dir / "sample_hr.png").convert("RGB")
sr_img = Image.open(output_dir / "sample_rrdb_sr.png").convert("RGB")

bicubic_img = lr_img.resize(hr_img.size, Image.BICUBIC)

plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.imshow(lr_img)
plt.title(f"LR\n{lr_img.size}")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(bicubic_img)
plt.title("Bicubic")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(sr_img)
plt.title("RRDB-LINF")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(hr_img)
plt.title("HR")
plt.axis("off")

plt.tight_layout()

save_path = "outputs/rrdb_minimal/rrdb_comparison.png"
plt.savefig(save_path, dpi=150)
print(f"저장 완료: {save_path}")
