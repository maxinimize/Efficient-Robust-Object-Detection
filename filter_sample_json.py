import json
import os
from pathlib import Path
import sys

if len(sys.argv) < 5:
    print("Usage: python filter_sample_json.py val/image/dir val/output/json/path train/image/dir train/output/json/path")
    sys.exit(1)

# --- Configuration ---
# -- Validation --
# Path to the original validation JSON annotation file
val_json_path = Path("data/COCO2017/annotations/instances_val2017.json") # <-- path to your JSON file

# Path to the folder containing the validation images you want to keep
val_image_dir = Path(sys.argv[1]) # <-- path to your image directory

# Path to save the new generated validation JSON file
val_output_json_path = Path(sys.argv[2]) # <-- path to your output JSON file

# -- Training --
# Path to the original training JSON annotation file
train_json_path = Path("data/COCO2017/annotations/instances_train2017.json") # <--  path to your JSON file

# Path to the folder containing the training images you want to keep
train_image_dir = Path(sys.argv[3]) # <-- path to your image directory

# Path to save the new generated training JSON file
train_output_json_path = Path(sys.argv[4]) # <-- path to your output JSON file

# --- JSON filter function ---
def json_filter(original_json_path, image_dir, output_json_path):
    # 1. Get all image filenames in the image_dir directory
    try:
        image_filenames = {f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))}
        print(f"Found {len(image_filenames)} images in the '{image_dir}' directory.")
    except FileNotFoundError:
        print(f"Error: Directory '{image_dir}' not found. Please check the path.")
        exit()

    # 2. Load the original JSON file
    try:
        with open(original_json_path, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded original JSON file: '{original_json_path}'.")
    except FileNotFoundError:
        print(f"Error: JSON file '{original_json_path}' not found. Please check the path.")
        exit()


    # 3. Filter the "images" list and collect the image_ids to keep
    filtered_images = [img for img in data['images'] if img['file_name'] in image_filenames]
    kept_image_ids = {img['id'] for img in filtered_images}

    print(f"After filtering, {len(filtered_images)} image annotations will be kept.")

    # 4. Filter the "annotations" list based on the kept image_ids
    filtered_annotations = [ann for ann in data['annotations'] if ann['image_id'] in kept_image_ids]

    print(f"After filtering, {len(filtered_annotations)} annotations will be kept.")

    # 5. Reassign category ids to make classes 0-80 rather than the spaghetti it is originally
    for counter,i in enumerate(data['categories']):
        true_id = i['id']
        for ann in filtered_annotations:
            if ann['category_id'] == true_id: # replace true id with an expected id
                ann['category_id'] = counter # true id is in increasing order even if non-consecutive

    print(f"Category ids now conform")
        
    # x. Construct the new JSON data
    new_data = {
        'info': data.get('info', {}),
        'licenses': data.get('licenses', []),
        'images': filtered_images,
        'annotations': filtered_annotations,
        'categories': data.get('categories', [])
    }

    # 6. Write the new JSON data to a file
    with open(output_json_path, 'w') as f:
        json.dump(new_data, f, indent=4)

    print(f"Filtering complete! The new JSON file has been saved to: '{output_json_path}'")

# -- Script Starts --
if __name__ == '__main__':
    json_filter(val_json_path, val_image_dir, val_output_json_path)
    json_filter(train_json_path, train_image_dir, train_output_json_path)
