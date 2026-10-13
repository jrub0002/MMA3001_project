# import python libraries
import os     
import glob    
import json    
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
 
def main():
    # Root folder of the dataset
    dataset_dir = '/content/Project_MMA3001_33854920/Pork-Rasher-Error(Packaging)-1'
    print('=== Step 1: Verifying Dataset Directory ===')
    # Sanity check: stop early if the dataset folder is missing (e.g. Drive not mounted)
    if os.path.exists(dataset_dir):
        print(f'Dataset directory exists: {dataset_dir}')
    else:
        print(f'Error: Dataset directory does not exist at {dataset_dir}')
        return
 
    # Search for all _annotations.coco.json files
    # '**' with recursive=True looks through every subfolder (typically train/valid/test)
    annotation_files = glob.glob(os.path.join(dataset_dir, '**/*_annotations.coco.json'), recursive=True)
    print(f'Found {len(annotation_files)} annotation files:')
    for filepath in sorted(annotation_files):
        print(f' - {filepath}')
 
    print('\n=== Step 2: Loading COCO Annotations ===')
    # Dict mapping split name (e.g. 'train') -> parsed COCO JSON content
    splits_annotations = {}
    for filepath in annotation_files:
        # The split name is taken from the name of the folder containing the JSON file
        split_name = os.path.basename(os.path.dirname(filepath))
        with open(filepath, 'r', encoding='utf-8') as f:
            splits_annotations[split_name] = json.load(f)
 
    # Print a quick structural summary of each split to confirm the files loaded correctly
    for split_name, data in sorted(splits_annotations.items()):
        print(f'Split: {split_name.upper()}')
        print(f' - Available Keys: {list(data.keys())}')  # expected COCO keys: images, annotations, categories, ...
        if 'images' in data:
            print(f' - Number of Images: {len(data["images"])}')
        if 'annotations' in data:
            print(f' - Number of Annotations: {len(data["annotations"])}')  # = number of bounding boxes
        if 'categories' in data:
            print(f' - Categories: {[cat["name"] for cat in data["categories"]]}')
        print('-' * 40)
 
    print('\n=== Step 3: Analyzing Class Distribution ===')
    # Build a per-split lookup of category id -> name
    category_maps = {}
    for split_name, data in splits_annotations.items():
        category_maps[split_name] = {cat['id']: cat['name'] for cat in data['categories']}
 
    # Union of all category names across splits, so every split is counted on the same set of classes
    all_categories = sorted(list(set(cat_name for split in category_maps.values() for cat_name in split.values())))
    # Initialise a zero count for every (split, category) pair
    split_counts = {split: {cat_name: 0 for cat_name in all_categories} for split in splits_annotations.keys()}
 
    # Count how many bounding boxes belong to each category in each split
    for split_name, data in splits_annotations.items():
        cat_map = category_maps[split_name]
        for ann in data['annotations']:
            cat_id = ann['category_id']
            cat_name = cat_map[cat_id]  # translate numeric id into a readable class name
            split_counts[split_name][cat_name] += 1
 
    # Rows = categories, columns = splits
    df_dist = pd.DataFrame(split_counts)
    print('--- Bounding Box Counts per Category across Splits ---')
    print(df_dist)
 
    # Plot distribution
    # Grouped bar chart: one group per category, one bar per split.
    # Note: the colour list has 3 entries (train/valid/test)
    ax = df_dist.plot(kind='bar', figsize=(10, 6), width=0.8, color=['#1f77b4', '#aec7e8', '#ff7f0e'])
    plt.title('Category Distribution Across Splits (Bounding Box Counts)')
    plt.xlabel('Category Name')
    plt.ylabel('Number of Annotations')
    plt.xticks(rotation=45, ha='right')  # tilt labels so long class names stay readable
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Dataset Splits')
    plt.tight_layout()
    # Save the figure for the verification report (the 'verification' folder must already exist)
    plot_path = '/content/Project_MMA3001_33854920/verification/class_distribution.png'
    plt.savefig(plot_path)
    plt.close()  # free memory / avoid showing the figure twice in notebooks
    print(f'Distribution plot saved to {plot_path}')
 
    print('\n=== Step 4: Analyzing Aspect Ratios and Relative Bounding Box Sizes ===')
    # Collect per-box statistics across ALL splits combined
    all_aspect_ratios = []   # width / height of each box
    all_relative_sizes = []  # box area as a fraction of its image's area
 
    for split_name, data in splits_annotations.items():
        # Lookup of image id -> (width, height) so each box can be compared to its parent image
        image_map = {img['id']: (img['width'], img['height']) for img in data['images']}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in image_map:
                continue  # skip orphan annotations that reference a missing image
            img_w, img_h = image_map[img_id]
            img_area = img_w * img_h
 
            # COCO bbox format is [x_min, y_min, width, height], so width/height are indices 2 and 3
            bbox = ann['bbox']
            bbox_w, bbox_h = bbox[2], bbox[3]
            bbox_area = bbox_w * bbox_h
 
            # Guard against division by zero from degenerate boxes or invalid image sizes
            if bbox_h > 0 and img_area > 0:
                aspect_ratio = bbox_w / bbox_h        # >1 = wide box, <1 = tall box
                relative_size = bbox_area / img_area  # 0-1; tiny values = small objects
                all_aspect_ratios.append(aspect_ratio)
                all_relative_sizes.append(relative_size)
 
    df_stats = pd.DataFrame({
        'Aspect Ratio': all_aspect_ratios,
        'Relative Size': all_relative_sizes
    })
 
    # Summary stats (count, mean, std, min, quartiles, 90th percentile, max)
    # help spot outliers and tune model settings such as anchor shapes and input size
    print('--- Summary Statistics for Bounding Boxes ---')
    print(df_stats.describe(percentiles=[0.25, 0.5, 0.75, 0.9]))
 
# Run main() only when the script is executed directly (not when imported as a module)
if __name__ == "__main__":
    main()
