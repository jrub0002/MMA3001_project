import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    dataset_dir = '/content/Project_MMA3001_33854920/Pork-Rasher-Error(Packaging)-1'
    print('=== Step 1: Verifying Dataset Directory ===')
    if os.path.exists(dataset_dir):
        print(f'Dataset directory exists: {dataset_dir}')
    else:
        print(f'Error: Dataset directory does not exist at {dataset_dir}')
        return

    # Search for all _annotations.coco.json files
    annotation_files = glob.glob(os.path.join(dataset_dir, '**/*_annotations.coco.json'), recursive=True)
    print(f'Found {len(annotation_files)} annotation files:')
    for filepath in sorted(annotation_files):
        print(f' - {filepath}')

    print('\n=== Step 2: Loading COCO Annotations ===')
    splits_annotations = {}
    for filepath in annotation_files:
        split_name = os.path.basename(os.path.dirname(filepath))
        with open(filepath, 'r', encoding='utf-8') as f:
            splits_annotations[split_name] = json.load(f)

    for split_name, data in sorted(splits_annotations.items()):
        print(f'Split: {split_name.upper()}')
        print(f' - Available Keys: {list(data.keys())}')
        if 'images' in data:
            print(f' - Number of Images: {len(data["images"])}')
        if 'annotations' in data:
            print(f' - Number of Annotations: {len(data["annotations"])}')
        if 'categories' in data:
            print(f' - Categories: {[cat["name"] for cat in data["categories"]]}')
        print('-' * 40)

    print('\n=== Step 3: Analyzing Class Distribution ===')
    category_maps = {}
    for split_name, data in splits_annotations.items():
        category_maps[split_name] = {cat['id']: cat['name'] for cat in data['categories']}

    all_categories = sorted(list(set(cat_name for split in category_maps.values() for cat_name in split.values())))
    split_counts = {split: {cat_name: 0 for cat_name in all_categories} for split in splits_annotations.keys()}

    for split_name, data in splits_annotations.items():
        cat_map = category_maps[split_name]
        for ann in data['annotations']:
            cat_id = ann['category_id']
            cat_name = cat_map[cat_id]
            split_counts[split_name][cat_name] += 1

    df_dist = pd.DataFrame(split_counts)
    print('--- Bounding Box Counts per Category across Splits ---')
    print(df_dist)

    # Plot distribution
    ax = df_dist.plot(kind='bar', figsize=(10, 6), width=0.8, color=['#1f77b4', '#aec7e8', '#ff7f0e'])
    plt.title('Category Distribution Across Splits (Bounding Box Counts)')
    plt.xlabel('Category Name')
    plt.ylabel('Number of Annotations')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Dataset Splits')
    plt.tight_layout()
    plot_path = '/content/Project_MMA3001_33854920/verification/class_distribution.png'
    plt.savefig(plot_path)
    plt.close()
    print(f'Distribution plot saved to {plot_path}')

    print('\n=== Step 4: Analyzing Aspect Ratios and Relative Bounding Box Sizes ===')
    all_aspect_ratios = []
    all_relative_sizes = []

    for split_name, data in splits_annotations.items():
        image_map = {img['id']: (img['width'], img['height']) for img in data['images']}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in image_map:
                continue
            img_w, img_h = image_map[img_id]
            img_area = img_w * img_h

            bbox = ann['bbox']
            bbox_w, bbox_h = bbox[2], bbox[3]
            bbox_area = bbox_w * bbox_h

            if bbox_h > 0 and img_area > 0:
                aspect_ratio = bbox_w / bbox_h
                relative_size = bbox_area / img_area
                all_aspect_ratios.append(aspect_ratio)
                all_relative_sizes.append(relative_size)

    df_stats = pd.DataFrame({
        'Aspect Ratio': all_aspect_ratios,
        'Relative Size': all_relative_sizes
    })

    print('--- Summary Statistics for Bounding Boxes ---')
    print(df_stats.describe(percentiles=[0.25, 0.5, 0.75, 0.9]))

if __name__ == "__main__":
    main()