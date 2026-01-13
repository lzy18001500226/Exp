"""Filter out samples containing target classes from NPZ index."""
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

def filter_classes(input_json: str, output_json: str, exclude_classes: list):
    """Remove samples containing any of the exclude_classes."""
    with open(input_json, 'r') as f:
        paths = json.load(f)
    
    print(f"Original samples: {len(paths)}")
    print(f"Excluding classes: {exclude_classes}")
    
    filtered = []
    excluded_count = 0
    
    for p in tqdm(paths, desc="Filtering"):
        try:
            z = np.load(p, allow_pickle=True)
            
            # Check if has bboxes
            if 'bboxes' not in z or z['bboxes'].size == 0:
                # No objects, keep it (might be background)
                filtered.append(p)
                continue
            
            bboxes = z['bboxes']
            
            # Extract class ids (handle both [cls,cx,cy,w,h] and [x1,y1,x2,y2,cls] formats)
            if bboxes.shape[1] == 5:
                # Check which format: if max value > 2, likely pixel coords [x1,y1,x2,y2,cls]
                if bboxes.max() > 2.0:
                    classes = bboxes[:, 4]  # last column is class
                else:
                    classes = bboxes[:, 0]  # first column is class (normalized format)
            else:
                print(f"[warn] Unexpected bbox shape {bboxes.shape} in {p}")
                continue
            
            # Check if contains any excluded class
            has_excluded = any(int(c) in exclude_classes for c in classes)
            
            if not has_excluded:
                filtered.append(p)
            else:
                excluded_count += 1
        
        except Exception as e:
            print(f"[error] Failed to process {p}: {e}")
            continue
    
    print(f"Filtered samples: {len(filtered)}")
    print(f"Excluded samples: {excluded_count}")
    
    with open(output_json, 'w') as f:
        json.dump(filtered, f, indent=2)
    
    print(f"Saved to {output_json}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Filter out samples containing specified classes')
    parser.add_argument('--classes', type=str, default='2,7,20',
                        help='Comma-separated list of class IDs to exclude (default: 2,7,20)')
    parser.add_argument('--train-json', type=str, default='data_preprocessed_train.json',
                        help='Input training JSON')
    parser.add_argument('--val-json', type=str, default='data_preprocessed_val.json',
                        help='Input validation JSON')
    parser.add_argument('--output-train', type=str, default='data_preprocessed_train_no_weak.json',
                        help='Output training JSON')
    parser.add_argument('--output-val', type=str, default='data_preprocessed_val_no_weak.json',
                        help='Output validation JSON')
    
    args = parser.parse_args()
    
    # Parse class IDs
    exclude_classes = [int(c.strip()) for c in args.classes.split(',')]
    
    print(f"Excluding classes: {exclude_classes}")
    print()
    
    # Train set
    print("=" * 60)
    print("Processing training set...")
    print("=" * 60)
    filter_classes(
        args.train_json,
        args.output_train,
        exclude_classes=exclude_classes
    )
    
    print()
    
    # Val set
    print("=" * 60)
    print("Processing validation set...")
    print("=" * 60)
    filter_classes(
        args.val_json,
        args.output_val,
        exclude_classes=exclude_classes
    )
