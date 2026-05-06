"""
Считаем картинки в dataset2-master по классам.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC = PROJECT_ROOT / 'dataset' / 'dataset2-master' / 'dataset2-master' / 'images'

CLASSES = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

for split in ['TRAIN', 'TEST', 'TEST_SIMPLE']:
    print(f"\n{split}:")
    split_dir = SRC / split
    if not split_dir.exists():
        print(f"  (нет такой папки)")
        continue
    for cls in CLASSES:
        cls_dir = split_dir / cls
        if cls_dir.exists():
            count = len(list(cls_dir.glob('*.jp*g')))  # jpeg или jpg
            print(f"  {cls:12s}: {count}")
        else:
            print(f"  {cls:12s}: (нет папки)")