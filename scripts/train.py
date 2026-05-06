"""
Обучение Op-YOLOv8 на датасете BCCD.
Запуск: правый клик в редакторе → Run 'train'
"""
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent

def main():
    # Путь к нашему YAML-конфигу архитектуры
    model_yaml = PROJECT_ROOT / 'ultralytics' / 'ultralytics' / 'cfg' / 'models' / 'v8' / 'op_yolov8.yaml'
    data_yaml = PROJECT_ROOT / 'configs' / 'bccd.yaml'

    print(f"Модель: {model_yaml}")
    print(f"Данные: {data_yaml}")

    model = YOLO(str(model_yaml))

    results = model.train(
        data=str(data_yaml),
        epochs=100,
        imgsz=640,
        batch=64,                    # уменьши до 16 или 32, если кончится память GPU
        device=0,                    # GPU 0; 'cpu' если без GPU
        # Гиперпараметры (как в статье)
        lr0=0.01,
        weight_decay=0.0005,
        momentum=0.937,
        # Аугментация
        degrees=15,
        fliplr=0.5,
        flipud=0.5,
        hsv_v=0.4,
        # Сохранение
        project=str(PROJECT_ROOT / 'runs'),
        name='op_yolov8_bccd',
        exist_ok=False,              # не перезаписывать предыдущие запуски
    )

    print("\nОбучение завершено.")
    print(f"Лучший вес: {PROJECT_ROOT / 'runs' / 'op_yolov8_bccd' / 'weights' / 'best.pt'}")


if __name__ == '__main__':
    main()