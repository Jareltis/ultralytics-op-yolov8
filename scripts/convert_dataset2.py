"""
Конвертация classification dataset (dataset2-master) в YOLO detection format.
Каждой картинке присваивается bbox [0.5, 0.5, 0.95, 0.95] — почти весь кадр.

Разбиение:
- TEST из исходного датасета  -> наш test (20%)
- TRAIN из исходного датасета -> делим на наши train (87.5%) и val (12.5%)
  что в общем даёт ~70% train и ~10% val
"""
import shutil
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC = PROJECT_ROOT / 'dataset' / 'dataset2-master' / 'dataset2-master' / 'images'
DST = PROJECT_ROOT / 'datasets' / 'BCCD'

CLASSES_ORDERED = ['Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']
SRC_TO_IDX = {
    'EOSINOPHIL': 0,
    'LYMPHOCYTE': 1,
    'MONOCYTE':   2,
    'NEUTROPHIL': 3,
}

SEED = 42
TRAIN_RATIO_OF_SRC_TRAIN = 0.875  # 87.5% от исходного TRAIN -> наш train; остальное -> val

# YOLO bbox для всей клетки в центре картинки
BBOX_CX, BBOX_CY = 0.5, 0.5
BBOX_W, BBOX_H = 0.95, 0.95


def collect_images(folder):
    """Собираем все .jpeg/.jpg/.png из папки."""
    return list(folder.glob('*.jpeg')) + list(folder.glob('*.jpg')) + list(folder.glob('*.png'))


def write_sample(img_path, class_idx, dst_split):
    """Копируем картинку и создаём txt-аннотацию."""
    dst_img = DST / dst_split / 'images' / img_path.name
    dst_label = DST / dst_split / 'labels' / f'{img_path.stem}.txt'
    shutil.copy2(img_path, dst_img)
    dst_label.write_text(f"{class_idx} {BBOX_CX} {BBOX_CY} {BBOX_W} {BBOX_H}")


def main():
    print(f"Source: {SRC}")
    print(f"Output: {DST}\n")

    # Создаём выходные папки
    for split in ['train', 'val', 'test']:
        (DST / split / 'images').mkdir(parents=True, exist_ok=True)
        (DST / split / 'labels').mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    counts = {'train': [0]*4, 'val': [0]*4, 'test': [0]*4}

    # Обработка TEST -> test
    for src_cls, idx in SRC_TO_IDX.items():
        for img in collect_images(SRC / 'TEST' / src_cls):
            write_sample(img, idx, 'test')
            counts['test'][idx] += 1

    # Обработка TRAIN -> train + val (внутри каждого класса)
    for src_cls, idx in SRC_TO_IDX.items():
        images = collect_images(SRC / 'TRAIN' / src_cls)
        random.shuffle(images)
        n_train = int(len(images) * TRAIN_RATIO_OF_SRC_TRAIN)
        for img in images[:n_train]:
            write_sample(img, idx, 'train')
            counts['train'][idx] += 1
        for img in images[n_train:]:
            write_sample(img, idx, 'val')
            counts['val'][idx] += 1

    print("Распределение классов:")
    for split in ['train', 'val', 'test']:
        line = f"  {split:5s}: total={sum(counts[split]):4d}  |  "
        line += ", ".join(f"{CLASSES_ORDERED[i]}={counts[split][i]}" for i in range(4))
        print(line)
    print(f"\n✓ Готово. Результаты в {DST}")


if __name__ == '__main__':
    main()