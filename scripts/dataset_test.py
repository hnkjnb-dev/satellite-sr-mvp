from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SRDataset(Dataset):

    def __init__(self, csv_path):

        self.df = pd.read_csv(csv_path)

        self.transform = transforms.ToTensor()

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        hr_path = row["hr_path"]
        lr_path = row["lr_path"]

        # 이미지 로드
        hr_img = Image.open(hr_path).convert("RGB")
        lr_img = Image.open(lr_path).convert("RGB")

        # Tensor 변환
        hr_tensor = self.transform(hr_img)
        lr_tensor = self.transform(lr_img)

        return {
            "hr": hr_tensor,
            "lr": lr_tensor,
            "filename": row["filename"]
        }


# Dataset 생성
dataset = SRDataset("metadata/train.csv")

print(f"Dataset 크기: {len(dataset)}")

# 첫 샘플 확인
sample = dataset[0]

print("\n첫 번째 샘플")
print(f"파일명: {sample['filename']}")
print(f"LR shape: {sample['lr'].shape}")
print(f"HR shape: {sample['hr'].shape}")

print(f"\nLR min/max: {sample['lr'].min()} / {sample['lr'].max()}")
print(f"HR min/max: {sample['hr'].min()} / {sample['hr'].max()}")