import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(str(SRC_DIR))

print("PROJECT_ROOT:", PROJECT_ROOT)
print("SRC_DIR:", SRC_DIR)

try:
    import linf_sr
    print("linf_sr import 성공")

    from linf_sr import model
    print("linf_sr.model import 성공")

    names = [name for name in dir(model) if not name.startswith("_")]
    print("model.py 안의 주요 이름:")
    print(names)

except Exception as e:
    print("RRDB-LINF import 실패")
    print(type(e).__name__, e)