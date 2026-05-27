import json
from pathlib import Path

DATASET_DIR = Path("final_dataset")

BAD_CLASS_NAMES = {
    "f1-sponsors-dataset",
    "F1-Sponsors-Dataset",
    "f1_sponsors_dataset"
}

def clean_coco_file(coco_path: Path):
    print("\nFile:", coco_path)

    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    print("Before:")
    for cat in sorted(coco["categories"], key=lambda x: x["id"]):
        print(cat["id"], cat["name"])

    bad_ids = {
        cat["id"]
        for cat in coco["categories"]
        if cat["name"] in BAD_CLASS_NAMES
    }

    if not bad_ids:
        print("No bad class found.")
        return

    bad_ann_count = sum(
        ann["category_id"] in bad_ids
        for ann in coco["annotations"]
    )

    print("Bad class ids:", bad_ids)
    print("Annotations with bad class:", bad_ann_count)

    coco["annotations"] = [
        ann for ann in coco["annotations"]
        if ann["category_id"] not in bad_ids
    ]

    old_categories = [
        cat for cat in coco["categories"]
        if cat["id"] not in bad_ids
    ]

    old_to_new = {}
    new_categories = []

    for new_id, cat in enumerate(sorted(old_categories, key=lambda x: x["id"])):
        old_id = cat["id"]
        old_to_new[old_id] = new_id
        cat["id"] = new_id
        new_categories.append(cat)

    for ann in coco["annotations"]:
        ann["category_id"] = old_to_new[ann["category_id"]]

    coco["categories"] = new_categories

    with open(coco_path, "w", encoding="utf-8") as f:
        json.dump(coco, f)

    print("After:")
    for cat in coco["categories"]:
        print(cat["id"], cat["name"])


for split in ["train", "valid", "test"]:
    coco_path = DATASET_DIR / split / "_annotations.coco.json"

    if coco_path.exists():
        clean_coco_file(coco_path)
    else:
        print("Not found:", coco_path)