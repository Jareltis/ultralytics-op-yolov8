"""
Inference на нескольких картинках из test-сплита.
Сохраняет визуализации с bbox'ами в runs/test_predict/
"""
from pathlib import Path
import random
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent

def main():
    weights = PROJECT_ROOT / 'runs' / 'test_run' / 'weights' / 'best.pt'
    test_images = PROJECT_ROOT / 'datasets' / 'BCCD' / 'test' / 'images'

    model = YOLO(str(weights))

    # Выбираем по 3 случайные картинки каждого класса
    # Имена файлов в этом датасете не содержат класса, поэтому просто берём 12 случайных
    all_images = list(test_images.glob('*.jpeg')) + list(test_images.glob('*.jpg'))
    random.seed(42)
    sample = random.sample(all_images, 12)

    print(f"Запуск inference на 12 картинках...")

    results = model.predict(
        source=sample,
        imgsz=640,
        conf=0.25,                     # порог уверенности
        device=0,
        save=True,                     # сохранить картинки с нарисованными bbox'ами
        project=str(PROJECT_ROOT / 'runs'),
        name='test_predict',
        exist_ok=True,
    )

    print(f"\n✓ Готово. Результаты сохранены в {PROJECT_ROOT / 'runs' / 'test_predict'}")
    print("Открой эту папку и посмотри картинки — на них должны быть bbox'ы с подписями классов.")

if __name__ == '__main__':
    main()