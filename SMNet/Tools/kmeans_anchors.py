import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import KMeans

def iou(box, clusters):
    """
    Calculate IoU between a box and clusters.
    box: tuple or array, (w, h)
    clusters: numpy array, (k, 2) where each row is (w, h)
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_

def avg_iou(boxes, clusters):
    """
    Calculate the average IoU between all boxes and their closest cluster centroid.
    """
    return np.mean([np.max(iou(box, clusters)) for box in boxes])

def load_boxes_from_jsons(label_roots, img_size=512):
    boxes = []
    
    paths = []
    for root in label_roots:
        p = Path(root)
        if not p.exists():
            print(f"Warning: {root} does not exist.")
            continue
        # Recursively find all .json files
        paths.extend(list(p.rglob("*.json")))
    
    print(f"Found {len(paths)} JSON files. Loading boxes...")
    
    for json_path in tqdm(paths):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Get image dimensions, default to 512 if missing (though they should be there)
            W = data.get("imageWidth", 512)
            H = data.get("imageHeight", 512)
            
            # Scale factor to target img_size (512)
            # Assuming we resize the image to img_size x img_size
            # If aspect ratio is preserved, scaling might be different for w and h
            # But usually for 512x512 training, we resize both dims.
            scale_w = img_size / W
            scale_h = img_size / H
            
            for shape in data.get("shapes", []):
                if shape.get("shape_type") not in ["rectangle", "Rect", "rect"]:
                    continue
                
                pts = shape.get("points", [])
                if len(pts) < 2:
                    continue
                
                x1, y1 = pts[0]
                x2, y2 = pts[1]
                
                w = abs(x2 - x1) * scale_w
                h = abs(y2 - y1) * scale_h
                
                if w > 0 and h > 0:
                    boxes.append([w, h])
                    
        except Exception as e:
            # print(f"Error reading {json_path}: {e}")
            pass
            
    return np.array(boxes)

def main():
    parser = argparse.ArgumentParser(description="K-Means Clustering for Anchor Box Generation")
    parser.add_argument("--label-dirs", nargs='+', default=[r"D:\Exp\FCSLabel", r"D:\Exp\VTSLabel"], help="Directories containing label JSONs")
    parser.add_argument("--num-clusters", type=int, default=9, help="Number of anchor clusters (e.g., 9 for 3 scales * 3 anchors)")
    parser.add_argument("--img-size", type=int, default=512, help="Target image size for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f"Scanning directories: {args.label_dirs}")
    boxes = load_boxes_from_jsons(args.label_dirs, args.img_size)
    
    if len(boxes) == 0:
        print("No boxes found!")
        return

    print(f"Loaded {len(boxes)} boxes.")
    
    print(f"Running K-Means with k={args.num_clusters}...")
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=args.seed, n_init=10)
    kmeans.fit(boxes)
    
    anchors = kmeans.cluster_centers_
    
    # Sort by area
    areas = anchors[:, 0] * anchors[:, 1]
    sorted_indices = np.argsort(areas)
    anchors = anchors[sorted_indices]
    
    print("\nGenerated Anchors (Width x Height):")
    for w, h in anchors:
        print(f"{w:.2f} x {h:.2f}")
        
    accuracy = avg_iou(boxes, anchors)
    print(f"\nAverage IoU: {accuracy:.4f}")
    
    # Format for config
    anchor_str = ", ".join([f"{int(w)}x{int(h)}" for w, h in anchors])
    print(f"\nConfig String (copy this):")
    print(f"{anchor_str}")
    
    # Suggest split for 3 levels (Small, Medium, Large)
    if args.num_clusters == 9:
        print("\nSuggested Split for P2/P3/P4 (3 scales):")
        print(f"P2 (Small):  {', '.join([f'{int(w)}x{int(h)}' for w, h in anchors[0:3]])}")
        print(f"P3 (Medium): {', '.join([f'{int(w)}x{int(h)}' for w, h in anchors[3:6]])}")
        print(f"P4 (Large):  {', '.join([f'{int(w)}x{int(h)}' for w, h in anchors[6:9]])}")

if __name__ == "__main__":
    main()
