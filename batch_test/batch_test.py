import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from image_augmentation import jitter_image_random, split_image_diagonal_random
from cv_utils import Stitcher, calculate_ssim

import json
import time
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

import pandas as pd
from PIL import Image

    
def make_json_serializable(obj):
    """Converts NumPy types to Python natives for JSON"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (list, tuple)):
        return type(obj)(make_json_serializable(x) for x in obj)
    elif isinstance(obj, dict):
        return {make_json_serializable(k): make_json_serializable(v) for k, v in obj.items()}
    return obj


def run_batch_test(image_folder, output_json="ssim_results.json", rot_limit=5, trans_limit=3,
                  persp_limit=0.02, bright_factor=1.0, overlap_pct=0.15, num_images=100000):
    """
    Performs batch test with multiple images and saves SSIM scores in JSON
    """
    stitcher = Stitcher()
    image_folder = Path(image_folder)
    image_files = list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png")) + list(image_folder.glob("*.jpeg"))
    
    if len(image_files) < num_images:
        print(f"Warning: Only {len(image_files)} images found, instead of {num_images}")
        image_files = image_files[:num_images]
    else:
        image_files = random.sample(image_files, num_images)
    
    results = {
        "config": {
            "rotation_limit": rot_limit,
            "translation_limit": trans_limit,
            "perspective_limit": persp_limit,
            "brightness_factor": bright_factor,
            "overlap_pct": overlap_pct,
            "num_images": len(image_files),
            "device": str(stitcher.device),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "results": []
    }
    
    successful_stitches = 0
    failed_stitches = 0
    
    print(f"Start batch test with {len(image_files)} images...")
    
    for i, img_path in enumerate(tqdm(image_files, desc="Processing images")):
        try:
            original = Image.open(img_path).convert("RGB")
            
            left, right, _ = split_image_diagonal_random(str(img_path), min_overlap_pct=overlap_pct)
            
            left = jitter_image_random(
                left,
                rot_limit=rot_limit,
                trans_limit=trans_limit,
                persp_limit=persp_limit,
                bright_factor=bright_factor
            )
            
            stitched, _, num_matches = stitcher.stitch(left, right)
            
            if stitched is not None:
                ssim_score, _ = calculate_ssim(original, stitched)
                success = True
                successful_stitches += 1
            else:
                ssim_score = None
                success = False
                failed_stitches += 1
            
            result = {
                "image_index": i,
                "filename": img_path.name,
                "filepath": str(img_path),
                "success": success,
                "ssim_score": ssim_score,
                "num_matches": num_matches,
                "image_size": original.size
            }
            results["results"].append(result)
            
        except Exception as e:
            print(f"Fehler bei {img_path.name}: {str(e)}")
            results["results"].append({
                "image_index": i,
                "filename": img_path.name,
                "filepath": str(img_path),
                "success": False,
                "ssim_score": None,
                "num_matches": 0,
                "image_size": None,
                "error": str(e)
            })
            failed_stitches += 1
    
    successful_results = [r for r in results["results"] if r["success"]]
    if successful_results:
        results["summary"] = {
            "total_images": len(image_files),
            "successful_stitches": successful_stitches,
            "failed_stitches": failed_stitches,
            "success_rate": f"{successful_stitches/len(image_files)*100:.1f}%",
            "mean_ssim": np.mean([r["ssim_score"] for r in successful_results]),
            "median_ssim": np.median([r["ssim_score"] for r in successful_results]),
            "ssim_std": np.std([r["ssim_score"] for r in successful_results]),
            "ssim_min": np.min([r["ssim_score"] for r in successful_results]),
            "ssim_max": np.max([r["ssim_score"] for r in successful_results]),
            "mean_matches": np.mean([r["num_matches"] for r in successful_results]),
            "median_matches": np.median([r["num_matches"] for r in successful_results]),
        }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(make_json_serializable(results), f, indent=2, ensure_ascii=False)
    
    df = pd.DataFrame(results["results"])
    csv_path = output_json.replace('.json', '.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"\n Batch test completed!")
    print(f" Results saved: {output_json}")
    print(f"success rate: {successful_stitches}/{len(image_files)} ({successful_stitches/len(image_files)*100:.1f}%)")
    if successful_results:
        print(f"SSIM - Mean: {results['summary']['mean_ssim']:.4f}, Median: {results['summary']['median_ssim']:.4f}")
        print(f"Matches - Mean: {results['summary']['mean_matches']:.1f}, Median: {results['summary']['median_matches']:.1f}")
    
    return results

if __name__ == "__main__":

    BATCH_IMAGE_FOLDER = "./batch_test/test"
    
    if os.path.exists(BATCH_IMAGE_FOLDER):
        print("found")
        results = run_batch_test(
            image_folder=BATCH_IMAGE_FOLDER,
            output_json="ssim_batch_test.json",
            rot_limit=5,
            trans_limit=3,
            persp_limit=0.02,
            bright_factor=1.0,
            overlap_pct=0.15,
            num_images=100000
        )
