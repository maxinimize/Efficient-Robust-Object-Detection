import json
import os

# --- Configuration ---
# Path to the original JSON annotation file
original_json_path = 'data/COCO2017/annotations/instances_val2017_modified.json' # <-- Please replace this with the path to your JSON file

# Path to the folder containing the images you want to keep
image_dir = 'data/COCO2017/images/valid_sample' 

# Path to save the new generated JSON file
output_json_path = 'data/COCO2017/annotations/instances_val2017_modified_sample.json'

# --- Script starts ---

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

# 5. Construct the new JSON data
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