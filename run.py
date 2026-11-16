"""
============================================================
BERTopic 课程设计一键运行
============================================================
"""

import subprocess
import sys
from pathlib import Path
import time

BASE_DIR = Path(__file__).parent
PYTHON = sys.executable

scripts = [
    "preprocess.py",
    "modeling.py",
    "lda_model.py",
    "visualization.py"
]

print("=" * 60)
print("BERTopic 一键运行")
print("=" * 60)

start = time.time()

for script in scripts:

    print(f"\n正在运行：{script}")
    print("-" * 60)

    result = subprocess.run(
        [PYTHON, str(BASE_DIR / script)]
    )

    if result.returncode != 0:
        print(f"\n❌ {script} 运行失败")
        break

    print(f"✅ {script} 完成")

print("\n" + "=" * 60)
print(f"全部完成，总耗时 {time.time()-start:.2f} 秒")
print("=" * 60)