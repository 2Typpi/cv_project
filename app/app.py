import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from image_augmentation import jitter_image, split_image_diagonal
from cv_utils import Stitcher, calculate_ssim

from PIL import Image
import numpy as np
import gradio as gr

stitcher = Stitcher()

def process_images(files, rot_limit, trans_limit_X, trans_limit_Y, persp_limit, bright_factor, overlap_pct):
    if files is None or len(files) == 0:
        return None, None, ""
    
    imgs = [Image.open(f.name) for f in files if True]
    if len(imgs) == 0:
        return None, None, ""
    
    if len(imgs) == 1:
        img = imgs[0]
        original = img.copy()
        f_path = files[0].name
        
        left, right, _ = split_image_diagonal(f_path, min_overlap_pct=overlap_pct)
        left = jitter_image(
            left,
            angle=rot_limit,
            tx=trans_limit_X,
            ty=trans_limit_Y,
            perspective_coeffs=persp_limit,
            brightness_factor=bright_factor
        )
        
        images_to_stitch = [left, right]
        stitched, feature_mapping, num_matches = stitcher.stitch(left, right)
        matches_viz = stitcher.visualize_matches(images_to_stitch, feature_mapping)
        
        if stitched is not None:
            result_img = Image.fromarray((stitched.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            ssim_score, _ = calculate_ssim(original, stitched)
            return result_img, matches_viz, f"SSIM: {ssim_score:.4f}"
        else:
            return img, matches_viz, "Stitching failed"
    
    elif len(imgs) == 2:
        img1, img2 = imgs[0], imgs[1]
        stitched, feature_mapping, num_matches = stitcher.stitch(img1, img2)
        matches_viz = stitcher.visualize_matches([img1, img2], feature_mapping)
        
        if stitched is not None:
            result_img = Image.fromarray((stitched.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            return result_img, matches_viz, ""
        else:
            return img1, matches_viz, "Stitching failed"
    
    return None, None, "Only support 1 or 2 images"

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## Automated Panorama Stitching")
    gr.Markdown("**1 Picture:** Split + Jitter + Stitch | **2 Pictures:** Direct Stitch")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_files = gr.File(label="Pictures (1 or 2)", file_types=["image"], file_count="multiple")
            
            with gr.Group("Jitter and Split"):
                rotation_slider = gr.Slider(0, 20, value=5, step=0.5, label="Rotation (°)")
                trans_slider_X = gr.Slider(0, 20, value=3, step=0.5, label="Translation X (px)")
                trans_slider_Y = gr.Slider(0, 20, value=3, step=0.5, label="Translation Y (px)")
                persp_slider = gr.Slider(0, 0.1, value=0.02, step=0.005, label="Perspective")
                bright_slider = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Brightness")
                overlap_slider = gr.Slider(0.05, 0.3, value=0.15, step=0.01, label="Overlap (%)")
            
            generate_btn = gr.Button("Stitch", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            stitched_gallery = gr.Image(label="Result")
            matches_gallery = gr.Image(label="Keypoints & Matches")
            stats_md = gr.Markdown(label="Stats")
    
    generate_btn.click(
        fn=process_images,
        inputs=[input_files, rotation_slider, trans_slider_X, trans_slider_Y, persp_slider, bright_slider, overlap_slider],
        outputs=[stitched_gallery, matches_gallery, stats_md]
    )

demo.launch(debug=True)
