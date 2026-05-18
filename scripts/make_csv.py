from pathlib import Path
import pandas as pd

hr_dir = Path("data/raw/hr")
lr_dir = Path("data/raw/lr")

rows = []

# HR 기준으로 순회
for hr_file in sorted(hr_dir.iterdir()):

    # 같은 이름 LR 찾기
    lr_file = lr_dir / hr_file.name

    # 존재 확인
    if lr_file.exists():

        rows.append({
            "hr_path": str(hr_file),
            "lr_path": str(lr_file),
            "filename": hr_file.name
        })

# DataFrame 생성
df = pd.DataFrame(rows)

# 저장
save_path = "metadata/train.csv"
df.to_csv(save_path, index=False)

print(f"CSV 저장 완료: {save_path}")
print(df.head())
print(f"\n총 pair 개수: {len(df)}")