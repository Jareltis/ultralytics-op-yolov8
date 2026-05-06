"""
Конвертация исходного BCCD (Shenggan) в YOLO-формат с 4 классами WBC.

Логика:
- Из XML берём bounding box'ы только для класса 'WBC' (RBC/Platelets отбрасываем)
- Тип WBC берём из labels.csv (одна метка на всё изображение)
- Изображения со смешанными метками или BASOPHIL пропускаем
- Делим на train/val/test = 70/10/20 как в статье
"""
import csv
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# ============ КОНФИГУРАЦИЯ ============
PROJECT_ROOT = Path(__file__).parent.parent
SOURCE_DIR = PROJECT_ROOT / 'dataset' / 'dataset-master' / 'dataset-master'
OUTPUT_DIR = PROJECT_ROOT / 'datasets' / 'BCCD'

XML_DIR = SOURCE_DIR / 'Annotations'
IMG_DIR = SOURCE_DIR / 'JPEGImages'
CSV_PATH = SOURCE_DIR / 'labels.csv'

# Маппинг класс → индекс (порядок как в статье)
CLASSES = ['Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']
CLASS_TO_IDX = {c.upper(): i for i, c in enumerate(CLASSES)}

SEED = 42  # для воспроизводимости разбиения
SPLIT_RATIOS = (0.70, 0.10, 0.20)  # train, val, test


def parse_csv(csv_path):
    """Читает labels.csv, возвращает {image_id: wbc_type или None для пропуска}."""
    image_to_type = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # пропускаем заголовок
        for row in reader:
            if len(row) < 3:
                continue
            try:
                image_id = int(row[1])
            except ValueError:
                continue
            category = row[2].strip().upper()
            # Пропускаем смешанные метки и BASOPHIL и LYMPHOCYTE с разными написаниями
            if ',' in category:
                image_to_type[image_id] = None  # смешанное — пропустить
                continue
            if category not in CLASS_TO_IDX:
                image_to_type[image_id] = None  # неизвестный класс
                continue
            image_to_type[image_id] = category
    return image_to_type


def parse_xml(xml_path):
    """Читает XML, возвращает (width, height, [список WBC bbox в формате (xmin, ymin, xmax, ymax)])."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text.strip()
        if name != 'WBC':
            continue  # пропускаем RBC и Platelets
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        boxes.append((xmin, ymin, xmax, ymax))
    return w, h, boxes


def to_yolo_format(w, h, xmin, ymin, xmax, ymax):
    """Конвертирует Pascal VOC bbox (xyxy в пикселях) в YOLO (xywh нормализованный)."""
    x_center = (xmin + xmax) / 2.0 / w
    y_center = (ymin + ymax) / 2.0 / h
    width = (xmax - xmin) / w
    height = (ymax - ymin) / h
    return x_center, y_center, width, height


def main():
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}\n")

    # Создаём выходные папки
    for split in ['train', 'val', 'test']:
        (OUTPUT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)

    # 1. Читаем csv
    print("Чтение labels.csv...")
    image_to_type = parse_csv(CSV_PATH)
    valid_count = sum(1 for v in image_to_type.values() if v is not None)
    print(f"  Всего записей: {len(image_to_type)}")
    print(f"  С валидной меткой WBC: {valid_count}\n")

    # 2. Собираем валидные изображения
    print("Парсинг XML и фильтрация...")
    valid_samples = []  # (image_path, xml_path, image_id, wbc_class_idx)
    skipped_no_csv = 0
    skipped_mixed = 0
    skipped_no_wbc = 0
    skipped_no_image = 0

    for xml_path in sorted(XML_DIR.glob('*.xml')):
        # Извлекаем номер из имени BloodImage_NNNNN.xml
        try:
            image_id = int(xml_path.stem.split('_')[1])
        except (IndexError, ValueError):
            continue

        # Проверяем csv
        wbc_type = image_to_type.get(image_id)
        if wbc_type is None:
            if image_id in image_to_type:
                skipped_mixed += 1
            else:
                skipped_no_csv += 1
            continue

        # Проверяем существование изображения
        img_path = IMG_DIR / f'{xml_path.stem}.jpg'
        if not img_path.exists():
            skipped_no_image += 1
            continue

        # Проверяем XML
        w, h, boxes = parse_xml(xml_path)
        if len(boxes) == 0:
            skipped_no_wbc += 1
            continue

        class_idx = CLASS_TO_IDX[wbc_type]
        valid_samples.append((img_path, xml_path, image_id, class_idx, w, h, boxes))

    print(f"  Пропущено (нет csv-метки):     {skipped_no_csv}")
    print(f"  Пропущено (смешанная метка):   {skipped_mixed}")
    print(f"  Пропущено (нет WBC в XML):     {skipped_no_wbc}")
    print(f"  Пропущено (нет файла jpg):     {skipped_no_image}")
    print(f"  Валидных образцов:             {len(valid_samples)}\n")

    # 3. Разбиение train/val/test
    random.seed(SEED)
    random.shuffle(valid_samples)
    n = len(valid_samples)
    n_train = int(n * SPLIT_RATIOS[0])
    n_val = int(n * SPLIT_RATIOS[1])
    splits = {
        'train': valid_samples[:n_train],
        'val': valid_samples[n_train:n_train + n_val],
        'test': valid_samples[n_train + n_val:],
    }
    for name, items in splits.items():
        print(f"  {name}: {len(items)} образцов")
    print()

    # 4. Подсчёт распределения классов
    print("Распределение классов:")
    for split_name, items in splits.items():
        counts = [0] * len(CLASSES)
        for *_, class_idx, _, _, _ in [(p, x, i, c, w, h, b) for p, x, i, c, w, h, b in items]:
            counts[class_idx] += 1
        line = f"  {split_name}: " + ", ".join(f"{CLASSES[i]}={counts[i]}" for i in range(len(CLASSES)))
        print(line)
    print()

    # 5. Копирование файлов и создание labels
    print("Копирование файлов...")
    for split_name, items in splits.items():
        for img_path, xml_path, image_id, class_idx, w, h, boxes in items:
            # Копируем картинку
            dst_img = OUTPUT_DIR / split_name / 'images' / img_path.name
            shutil.copy2(img_path, dst_img)

            # Создаём label-файл
            label_lines = []
            for xmin, ymin, xmax, ymax in boxes:
                x_c, y_c, bw, bh = to_yolo_format(w, h, xmin, ymin, xmax, ymax)
                label_lines.append(f"{class_idx} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")
            dst_label = OUTPUT_DIR / split_name / 'labels' / f'{img_path.stem}.txt'
            dst_label.write_text('\n'.join(label_lines))

    print(f"\n✓ Готово. Результаты в {OUTPUT_DIR}")


if __name__ == '__main__':
    main()