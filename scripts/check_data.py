from pathlib import Path

import rasterio
from PIL import Image


HR_DIR = Path("data/raw/hr")
LR_DIR = Path("data/raw/lr")


def check_with_rasterio(file_path: Path) -> bool:
    """GeoTIFF/TIFF 계열 파일 정보를 확인합니다."""
    try:
        with rasterio.open(file_path) as src:
            print("[rasterio로 열림]")
            print(f"파일명: {file_path.name}")
            print(f"크기(width x height): {src.width} x {src.height}")
            print(f"채널 수: {src.count}")
            print(f"dtype: {src.dtypes}")
            print(f"CRS: {src.crs}")
            print(f"Bounds: {src.bounds}")
            print(f"Transform: {src.transform}")
        return True
    except Exception as e:
        print(f"[rasterio 실패] {e}")
        return False


def check_with_pil(file_path: Path) -> bool:
    """PNG/JPG 등 일반 이미지 파일 정보를 확인합니다."""
    try:
        with Image.open(file_path) as img:
            print("[PIL로 열림]")
            print(f"파일명: {file_path.name}")
            print(f"크기(width x height): {img.size[0]} x {img.size[1]}")
            print(f"모드: {img.mode}")
            print(f"포맷: {img.format}")
        return True
    except Exception as e:
        print(f"[PIL 실패] {e}")
        return False


def check_folder(folder: Path, label: str) -> None:
    print("\n" + "=" * 70)
    print(f"{label} FILES")
    print("=" * 70)

    if not folder.exists():
        print(f"폴더가 없습니다: {folder}")
        return

    files = sorted([p for p in folder.iterdir() if p.is_file()])

    if not files:
        print(f"파일이 없습니다: {folder}")
        return

    print(f"파일 개수: {len(files)}")

    for file_path in files:
        print("\n" + "-" * 70)
        print(f"확인 중: {file_path}")

        opened = check_with_rasterio(file_path)

        if not opened:
            check_with_pil(file_path)


def main():
    check_folder(HR_DIR, "HR")
    check_folder(LR_DIR, "LR")


if __name__ == "__main__":
    main()