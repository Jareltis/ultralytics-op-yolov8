"""
Валидация обученной op_yolov8m модели на test-сплите BCCD.
"""
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent

def main():
    weights = PROJECT_ROOT / 'runs' / 'test_run' / 'weights' / 'best.pt'
    data_yaml = PROJECT_ROOT / 'configs' / 'bccd.yaml'

    print(f"Weights: {weights}")
    print(f"Data:    {data_yaml}\n")

    model = YOLO(str(weights))

    metrics = model.val(
        data=str(data_yaml),
        split='test',           # ← важно: test, а не val
        imgsz=640,
        batch=8,
        device=0,
        project=str(PROJECT_ROOT / 'runs'),
        name='test_eval',
        exist_ok=True,
    )

    print("\n=== Метрики на TEST-сплите ===")
    print(f"Precision (mean):  {metrics.box.mp:.4f}")
    print(f"Recall (mean):     {metrics.box.mr:.4f}")
    print(f"mAP50:             {metrics.box.map50:.4f}")
    print(f"mAP50-95:          {metrics.box.map:.4f}")

    print("\n=== По классам ===")
    class_names = ['Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']
    for i, name in enumerate(class_names):
        try:
            p = metrics.box.p[i] if hasattr(metrics.box, 'p') and len(metrics.box.p) > i else 0
            r = metrics.box.r[i] if hasattr(metrics.box, 'r') and len(metrics.box.r) > i else 0
            ap50 = metrics.box.ap50[i] if hasattr(metrics.box, 'ap50') and len(metrics.box.ap50) > i else 0
            ap = metrics.box.ap[i] if hasattr(metrics.box, 'ap') and len(metrics.box.ap) > i else 0
            print(f"{name:12s}: P={p:.3f}  R={r:.3f}  mAP50={ap50:.3f}  mAP50-95={ap:.3f}")
        except Exception as e:
            print(f"{name:12s}: данные недоступны ({e})")

    print(f"\nРезультаты сохранены в {PROJECT_ROOT / 'runs' / 'test_eval'}")
    print("Confusion matrix и графики там же.")

if __name__ == '__main__':
    main()