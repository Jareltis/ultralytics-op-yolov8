"""
Быстрая проверка - 3 эпохи на BCCD.
Цель: убедиться, что весь pipeline работает без ошибок.
"""
import sys
print("=== sys.path ===")
for p in sys.path:
    print(f"  {p}")

import ultralytics
print(f"\nultralytics.__file__ = {ultralytics.__file__}")

from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent

def main():
    model_yaml = PROJECT_ROOT / 'ultralytics' / 'ultralytics' / 'cfg' / 'models' / 'v8' / 'op_yolov8m.yaml'
    data_yaml = PROJECT_ROOT / 'configs' / 'bccd.yaml'

    print(f"Model YAML:  {model_yaml}")
    print(f"Data YAML:   {data_yaml}\n")

    model = YOLO(str(model_yaml).replace('op_yolov8', 'op_yolov8m'))

    results = model.train(
        data=str(data_yaml),
        epochs=3,                # ← всего 3 эпохи для проверки
        imgsz=640,
        batch=8,                 # маленький batch чтобы не упереться в GPU память
        device=0,                # 0 = GPU; 'cpu' если без GPU
        workers=4,               # параллельная загрузка данных
        project=str(PROJECT_ROOT / 'runs'),
        name='test_run',
        exist_ok=True,
    )
    print("\n✓ Pipeline works! Можно запускать полное обучение.")

if __name__ == '__main__':
    main()