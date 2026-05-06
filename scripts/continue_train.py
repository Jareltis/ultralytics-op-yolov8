"""
Продолжаем обучение с того места, где остановились.
17 эпох поверх существующих 3 = 20 эпох суммарно.
"""
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent

def main():
    weights = PROJECT_ROOT / 'runs' / 'test_run' / 'weights' / 'last.pt'
    data_yaml = PROJECT_ROOT / 'configs' / 'bccd.yaml'

    print(f"Загрузка весов: {weights}")
    model = YOLO(str(weights))

    results = model.train(
        data=str(data_yaml),
        epochs=20,                  # ← было 3, продолжаем до 20 (ещё 17 эпох)
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        cache='disk',
        amp=True,
        project=str(PROJECT_ROOT / 'runs'),
        name='test_run_20ep',
        exist_ok=True,
        resume=False,               # не resume, а fine-tune с этих весов
    )
    print("\n✓ Обучение продолжено.")

if __name__ == '__main__':
    main()