"""
Bayesian optimization гиперпараметров через Optuna.
"""
from pathlib import Path
import optuna
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_YAML = PROJECT_ROOT / 'ultralytics' / 'ultralytics' / 'cfg' / 'models' / 'v8' / 'op_yolov8.yaml'
DATA_YAML = PROJECT_ROOT / 'configs' / 'bccd.yaml'


def objective(trial):
    lr0 = trial.suggest_float('lr0', 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    batch = trial.suggest_categorical('batch', [16, 32, 64])

    model = YOLO(str(MODEL_YAML))
    results = model.train(
        data=str(DATA_YAML),
        epochs=30,                   # короткий пробный прогон
        imgsz=640,
        batch=batch,
        lr0=lr0,
        weight_decay=weight_decay,
        verbose=False,
        project=str(PROJECT_ROOT / 'runs' / 'tuning'),
        name=f'trial_{trial.number}',
        exist_ok=True,
    )
    return results.box.map50


def main():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    print("\n=== Результаты подбора ===")
    print(f"Лучшие параметры: {study.best_params}")
    print(f"Лучший mAP50:    {study.best_value:.3f}")


if __name__ == '__main__':
    main()