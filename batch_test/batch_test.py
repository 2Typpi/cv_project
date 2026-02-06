import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from image_augmentation import jitter_image_random, split_image_diagonal

import json
import time
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

import cv2
import torch
import kornia
import pandas as pd
from PIL import Image, ImageDraw
import torchvision.transforms as T
from transformers import AutoImageProcessor, AutoModel
from skimage.metrics import structural_similarity as ssim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

processor = AutoImageProcessor.from_pretrained("ETH-CVG/lightglue_superpoint")
model = AutoModel.from_pretrained("ETH-CVG/lightglue_superpoint")
model.to(device)

def draw_keypoints_and_matches(img_pil, keypoints, color=(0, 255, 0), radius=3):
    img_draw = ImageDraw.Draw(img_pil)
    kpts_np = keypoints.cpu().numpy()
    for kp in kpts_np:
        x, y = int(kp[0]), int(kp[1])
        img_draw.ellipse([x-radius, y-radius, x+radius, y+radius], outline=color, fill=color)
    return img_pil

def visualize_matches(images, feature_mapping):
    if len(feature_mapping) == 0:
        return images[0]
    matches_viz = processor.visualize_keypoint_matching(images, feature_mapping)
    return matches_viz[0]

def calculate_ssim(original_img, stitched_img, gray_scale=False):
    orig_np = np.array(original_img)
    
    stitch_np = stitched_img.cpu().detach().numpy()
    if stitch_np.ndim == 3:
        stitch_np = np.transpose(stitch_np, (1, 2, 0))
    stitch_np = cv2.resize(stitch_np, (orig_np.shape[1], orig_np.shape[0]))

    orig_np = orig_np.astype(np.float32) / 255.0 if orig_np.max() > 1.0 else orig_np.astype(np.float32)
    stitch_np = stitch_np.astype(np.float32) / 255.0 if stitch_np.max() > 1.0 else stitch_np.astype(np.float32)
    
    orig_np = np.dot(orig_np[..., :3], [0.2989, 0.5870, 0.1140]) if gray_scale else orig_np
    stitch_np = np.dot(stitch_np[..., :3], [0.2989, 0.5870, 0.1140]) if gray_scale else stitch_np
    
    score, diff = ssim(orig_np, stitch_np, full=True, data_range=1.0, channel_axis=None if gray_scale else 2)
    return score, diff

def stitch_images(img0_pil, img1_pil, output, device=device):
    to_tensor = T.ToTensor()
    image0 = to_tensor(img0_pil).to(device)
    image1 = to_tensor(img1_pil).to(device)

    pts0 = output["keypoints0"].float()
    pts1 = output["keypoints1"].float()

    num_matches = pts0.shape[0]
    print(f"DEBUG: {num_matches} Matches found!")
    
    if num_matches < 4:
        print("Insufficient matches for homography.")
        return None, num_matches

    p0_np = pts0.detach().cpu().numpy()
    p1_np = pts1.detach().cpu().numpy()

    H_np, mask = cv2.findHomography(
        p1_np, 
        p0_np, 
        method=cv2.USAC_MAGSAC, 
        ransacReprojThreshold=5.0, 
        confidence=0.999, 
        maxIters=100000
    )

    if H_np is None:
        print("Homography estimation failed.")
        return None, num_matches

    H = torch.from_numpy(H_np).to(device).float()

    c, h0, w0 = image0.shape
    _, h1, w1 = image1.shape

    corners1 = torch.tensor([[0., 0.], [float(w1), 0.], [float(w1), float(h1)], [0., float(h1)]], device=device)
    corners1_homo = torch.cat([corners1, torch.ones((4, 1), device=device)], dim=1).T
    warped_homo = H @ corners1_homo
    warped_corners1 = (warped_homo[:2] / warped_homo[2]).T
    
    all_coords = torch.cat([
        warped_corners1, 
        torch.tensor([[0., 0.], [float(w0), 0.], [float(w0), float(h0)], [0., float(h0)]], device=device)
    ], dim=0)
    
    min_xy = all_coords.min(dim=0).values
    max_xy = all_coords.max(dim=0).values

    translation = torch.eye(3, device=device)
    translation[0, 2] = -min_xy[0]
    translation[1, 2] = -min_xy[1]
    
    H_final = translation @ H
    out_size = (int(max_xy[1] - min_xy[1]), int(max_xy[0] - min_xy[0]))

    warped0 = kornia.geometry.transform.warp_perspective(
        image0.unsqueeze(0), translation.unsqueeze(0), dsize=out_size, align_corners=True
    ).squeeze(0)

    warped1 = kornia.geometry.transform.warp_perspective(
        image1.unsqueeze(0), H_final.unsqueeze(0), dsize=out_size, align_corners=True
    ).squeeze(0)

    mask0 = (warped0.abs().sum(dim=0, keepdim=True) > 1e-5).float()
    mask1 = (warped1.abs().sum(dim=0, keepdim=True) > 1e-5).float()
    
    stitched = (warped0 + warped1) / (mask0 + mask1 + 1e-8)     
    
    return stitched, num_matches

def feature_detection_mapping(images):
    inputs = processor(images, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    image_sizes = [[(image.height, image.width) for image in images]]
    outputs = processor.post_process_keypoint_matching(outputs, image_sizes, threshold=0.2)
    return outputs

    
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
            "device": str(device),
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
            
            left, right, _ = split_image_diagonal(str(img_path), min_overlap_pct=overlap_pct)
            
            left = jitter_image_random(
                left,
                rot_limit=rot_limit,
                trans_limit=trans_limit,
                persp_limit=persp_limit,
                bright_factor=bright_factor
            )
            
            images_to_stitch = [left, right]
            feature_mapping = feature_detection_mapping(images_to_stitch)
            
            stitch_result = stitch_images(left, right, feature_mapping[0], device=device)
            num_matches = 0
            stitched = None
            
            if stitch_result[0] is not None:
                stitched = stitch_result[0]
                num_matches = stitch_result[1]
                ssim_score, _ = calculate_ssim(original, stitched)
                success = True
                successful_stitches += 1
            else:
                num_matches = stitch_result[1]
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
