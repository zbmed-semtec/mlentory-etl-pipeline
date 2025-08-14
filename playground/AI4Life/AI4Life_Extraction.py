import json
from collections import Counter, defaultdict

# Configuration
INPUT_FILE   = 'experiment_metadata.json'
OUTPUT_FILES = {
    'model':       'models_only_.json',
    'dataset':     'datasets_only_.json',
    'application': 'applications_only_.json'
}
KNOWN_TYPES = set(OUTPUT_FILES.keys())

def main():
    # 1. Load the full JSON array
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. Count and display all encountered types
    type_list = [(entry.get('type') or '').strip().lower() for entry in data]
    type_counts = Counter(type_list)
    print("All types encountered:")
    for t, cnt in type_counts.items():
        label = t if t else '<<MISSING>>'
        print(f"  {label:12} : {cnt}")

    # 3. Group entries into known vs unknown
    groups  = defaultdict(list)
    unknown = defaultdict(list)
    for entry, t in zip(data, type_list):
        if t in KNOWN_TYPES:
            groups[t].append(entry)
        else:
            unknown[t].append(entry)

    # 4. Write out known-type files
    for t, out_path in OUTPUT_FILES.items():
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(groups.get(t, []), f, indent=4, ensure_ascii=False)

    # 5. Write unknown types to its own file
    with open('unknown_types.json', 'w', encoding='utf-8') as f:
        json.dump(unknown, f, indent=4, ensure_ascii=False)

    # 6. Summary report
    print("\nExport summary:")
    for t, out_path in OUTPUT_FILES.items():
        count = len(groups.get(t, []))
        print(f"  - {count:3d} entries of type '{t}' → {out_path}")
    total_unknown = sum(len(v) for v in unknown.values())
    print(f"  - {total_unknown:3d} entries of unknown/missing types → unknown_types.json")
    if unknown:
        unk_labels = [t or '<<MISSING>>' for t in unknown.keys()]
        print(f"    Unknown types found: {unk_labels}")

if __name__ == '__main__':
    main()
