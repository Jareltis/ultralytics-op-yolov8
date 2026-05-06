"""
Применяет histogram equalization (CLAHE) ко всем изображениям BCCD.
Запуск: правый клик в редакторе → Run 'preprocess'
"""
import cv2
from pathlib import Path

def preprocess_directory(input_dir: str, output_dir: str):
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    images = list(in_path.glob('*.jpg')) + list(in_path.glob('*.png'))
    print(f"Найдено {len(images)} изображений в {input_dir}")

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    for i, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Не удалось прочитать {img_path}")
            continue
        # CLAHE по каналу яркости в LAB-пространстве
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        cv2.imwrite(str(out_path / img_path.name), result)
        if i % 100 == 0:
            print(f"Обработано {i}/{len(images)}")

    print(f"Готово. Результаты в {output_dir}")


if __name__ == '__main__':
    base = Path(__file__).parent.parent / 'datasets' / 'BCCD' / 'images'
    for split in ['train', 'val', 'test']:
        print(f"\n=== {split} ===")
        preprocess_directory(
            input_dir=str(base / split),
            output_dir=str(base / f'{split}_eq'),
        )