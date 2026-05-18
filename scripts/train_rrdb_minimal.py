import sys
from pathlib import Path

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from linf_sr.model import RRDBLINF, make_coord_grid


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


def build_dummy_meta(batch_size, num_query, device):
    return torch.zeros(batch_size, num_query, 8, device=device)


def tensor_to_flat_pixels(hr):
    # hr: [B, 3, H, W] -> [B, H*W, 3]
    return hr.permute(0, 2, 3, 1).reshape(hr.shape[0], -1, 3)


def main():
    csv_path = "metadata/train.csv"

    output_dir = Path("outputs/rrdb_minimal")
    model_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 device: {device}")

    dataset = SRDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = RRDBLINF(
        in_channels=3,
        out_channels=3,
        feat_channels=48,
        num_blocks=3,
        growth_channels=24,
        decoder_hidden_dim=128,
        decoder_layers=3,
        meta_dim=8,
        encoder_meta_dim=6,
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 20
    num_train_queries = 8192

    print(f"학습 데이터 개수: {len(dataset)}")
    print("RRDB-LINF 최소 학습 시작")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for lr, hr, filename in dataloader:
            lr = lr.to(device)
            hr = hr.to(device)

            batch_size = lr.shape[0]
            target_h = hr.shape[2]
            target_w = hr.shape[3]
            total_pixels = target_h * target_w

            full_coord = make_coord_grid(target_h, target_w, device=device)
            full_coord = full_coord.unsqueeze(0).repeat(batch_size, 1, 1)

            hr_flat = tensor_to_flat_pixels(hr)

            # 전체 360,000픽셀 중 일부만 랜덤 샘플링해서 학습
            idx = torch.randperm(total_pixels, device=device)[:num_train_queries]

            coord = full_coord[:, idx, :]
            target = hr_flat[:, idx, :]
            meta = build_dummy_meta(batch_size, coord.shape[1], device)

            pred = model(
                lr=lr,
                coord=coord,
                meta=meta,
                query_chunk_size=4096,
            )

            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.6f}")

    save_model_path = model_dir / "rrdb_linf_minimal.pth"
    torch.save(model.state_dict(), save_model_path)
    print(f"모델 저장 완료: {save_model_path}")

    # 샘플 1장 전체 해상도 inference 저장
    model.eval()
    with torch.no_grad():
        lr, hr, filename = dataset[0]
        lr = lr.unsqueeze(0).to(device)
        hr = hr.unsqueeze(0).to(device)

        target_h = hr.shape[2]
        target_w = hr.shape[3]

        coord = make_coord_grid(target_h, target_w, device=device)
        coord = coord.unsqueeze(0)

        meta = build_dummy_meta(1, coord.shape[1], device)

        pred = model(
            lr=lr,
            coord=coord,
            meta=meta,
            query_chunk_size=4096,
        )

        sr = pred.view(1, target_h, target_w, 3).permute(0, 3, 1, 2)
        sr = sr.clamp(0, 1)

        save_image(lr.squeeze(0).cpu(), output_dir / "sample_lr.png")
        save_image(hr.squeeze(0).cpu(), output_dir / "sample_hr.png")
        save_image(sr.squeeze(0).cpu(), output_dir / "sample_rrdb_sr.png")

    print("샘플 결과 저장 완료")
    print("outputs/rrdb_minimal/sample_lr.png")
    print("outputs/rrdb_minimal/sample_hr.png")
    print("outputs/rrdb_minimal/sample_rrdb_sr.png")


if __name__ == "__main__":
    main()
