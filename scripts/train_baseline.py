from pathlib import Path

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image


class SRDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        lr_img = Image.open(row["lr_path"]).convert("RGB")
        hr_img = Image.open(row["hr_path"]).convert("RGB")

        lr = self.transform(lr_img)
        hr = self.transform(hr_img)

        return lr, hr, row["filename"]


class TinySRNet(nn.Module):
    def __init__(self, scale_factor=4):
        super().__init__()

        self.upsample = nn.Upsample(
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=False,
        )

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.net(x)
        return x


def main():
    csv_path = "metadata/train.csv"

    output_dir = Path("outputs/baseline")
    model_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 device: {device}")

    dataset = SRDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = TinySRNet(scale_factor=4).to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20

    print(f"학습 데이터 개수: {len(dataset)}")
    print("학습 시작")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for lr, hr, filename in dataloader:
            lr = lr.to(device)
            hr = hr.to(device)

            sr = model(lr)
            loss = criterion(sr, hr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), model_dir / "tiny_sr_baseline.pth")
    print("모델 저장 완료: models/tiny_sr_baseline.pth")

    model.eval()
    with torch.no_grad():
        lr, hr, filename = dataset[0]
        lr_batch = lr.unsqueeze(0).to(device)

        sr = model(lr_batch).cpu().squeeze(0)

        save_image(lr, output_dir / "sample_lr.png")
        save_image(hr, output_dir / "sample_hr.png")
        save_image(sr, output_dir / "sample_sr.png")

    print("샘플 결과 저장 완료")
    print("outputs/baseline/sample_lr.png")
    print("outputs/baseline/sample_hr.png")
    print("outputs/baseline/sample_sr.png")


if __name__ == "__main__":
    main()