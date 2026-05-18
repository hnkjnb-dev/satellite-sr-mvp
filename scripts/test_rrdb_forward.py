import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from linf_sr.model import RRDBLINF, make_coord_grid


def build_dummy_meta(batch_size, num_query, device):
    """
    RRDB-LINF는 coord 외에 meta 정보가 필요합니다.
    우선 forward 테스트용으로 8차원 dummy meta를 넣습니다.

    meta_dim = 8
    앞 2개: coord 관련 placeholder
    뒤 6개: encoder_meta 추론용 placeholder
    """
    return torch.zeros(batch_size, num_query, 8, device=device)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 device: {device}")

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

    model.eval()
    print("RRDBLINF 모델 생성 성공")

    lr_path = "data/raw/lr/C1_20250508013244_22921_00306979_2m_x1800_y4800.png"

    img = Image.open(lr_path).convert("RGB")
    transform = transforms.ToTensor()
    lr = transform(img).unsqueeze(0).to(device)

    batch_size = lr.shape[0]
    target_h = lr.shape[2] * 4
    target_w = lr.shape[3] * 4

    print(f"입력 LR shape: {lr.shape}")
    print(f"목표 HR 크기: {target_h} x {target_w}")

    coord = make_coord_grid(target_h, target_w, device=device)
    coord = coord.unsqueeze(0).repeat(batch_size, 1, 1)

    meta = build_dummy_meta(
        batch_size=batch_size,
        num_query=coord.shape[1],
        device=device,
    )

    print(f"coord shape: {coord.shape}")
    print(f"meta shape: {meta.shape}")

    with torch.no_grad():
        pred = model(
            lr=lr,
            coord=coord,
            meta=meta,
            query_chunk_size=4096,
        )

    print(f"모델 출력 pred shape: {pred.shape}")

    pred_img = pred.view(batch_size, target_h, target_w, 3)
    pred_img = pred_img.permute(0, 3, 1, 2)

    print(f"이미지 형태 변환 후 shape: {pred_img.shape}")
    print("RRDB-LINF forward 성공")


if __name__ == "__main__":
    main()