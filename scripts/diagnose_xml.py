"""
Диагностика: какие классы и в скольких XML-файлах присутствуют.
"""
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
XML_DIR = PROJECT_ROOT / 'dataset' / 'dataset-master' / 'dataset-master' / 'Annotations'

class_counts = Counter()           # сколько bbox'ов каждого класса всего
files_with_class = defaultdict(int)  # в скольких файлах встречается класс

xml_files = list(XML_DIR.glob('*.xml'))
print(f"Всего XML-файлов: {len(xml_files)}\n")

for xml_path in xml_files:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    classes_in_file = set()
    for obj in root.findall('object'):
        name = obj.find('name').text.strip()
        class_counts[name] += 1
        classes_in_file.add(name)
    for c in classes_in_file:
        files_with_class[c] += 1

print("Классы и количество bbox'ов:")
for cls, count in class_counts.most_common():
    print(f"  {cls:20s}: {count:5d} bbox'ов в {files_with_class[cls]} файлах")

print("\nПримеры файлов с разными классами:")
for cls in class_counts.keys():
    for xml_path in xml_files:
        tree = ET.parse(xml_path)
        for obj in tree.getroot().findall('object'):
            if obj.find('name').text.strip() == cls:
                print(f"  {cls:20s} -> {xml_path.name}")
                break
        else:
            continue
        break