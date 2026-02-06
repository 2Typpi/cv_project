import gradio as gr
from PIL import Image, ImageDraw, ImageEnhance
import torch
import kornia
from transformers import AutoImageProcessor, AutoModel
import torchvision.transforms as T
import numpy as np
import cv2
import random
import glob
from skimage.metrics import structural_similarity as ssim  # Hinzugefügt für SSIM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Lade LightGlue-Modell (genau wie Jupyter)
processor = AutoImageProcessor.from_pretrained("ETH-CVG/lightglue_superpoint")
model = AutoModel.from_pretrained("ETH-CVG/lightglue_superpoint")
model.to(device)

def draw_keypoints_and_matches(img_pil, keypoints, color=(0, 255, 0), radius=3):
    """Plots key points as colored circles on PIL image."""
    img_draw = ImageDraw.Draw(img_pil)
    kpts_np = keypoints.cpu().numpy()
    for kp in kpts_np:
        x, y = int(kp[0]), int(kp[1])
        img_draw.ellipse([x-radius, y-radius, x+radius, y+radius], outline=color, fill=color)
    return img_pil

def visualize_matches(images, feature_mapping):
    """Visualization """
    if len(feature_mapping) == 0:
        return images[0]
    
    matches_viz = processor.visualize_keypoint_matching(images, feature_mapping)
    return matches_viz[0]

def calculate_ssim(original_img, stitched_img, gray_scale=False):
    """
    Calculates the SSIM between the original PIL image and the stitched PIL image.
    """
    orig_np = np.array(original_img)
    
    stitch_np = stitched_img.cpu().detach().numpy()
    if stitch_np.ndim == 3:
        stitch_np = np.transpose(stitch_np, (1, 2, 0))
    # ensure imgs to have the same size 
    stitch_np = cv2.resize(stitch_np, (orig_np.shape[1], orig_np.shape[0]))

    print(orig_np.shape, stitch_np.shape)

    # ensure the data types and ranges match
    orig_np = orig_np.astype(np.float32) / 255.0 if orig_np.max() > 1.0 else orig_np.astype(np.float32)
    stitch_np = stitch_np.astype(np.float32) / 255.0 if stitch_np.max() > 1.0 else stitch_np.astype(np.float32)
    
    # conversion to gray scale
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

    print(f"DEBUG: {pts0.shape[0]} Matches found!")
    
    if pts0.shape[0] < 4:
        print("Insufficient matches for homography.")
        return None

    p0_np = pts0.detach().cpu().numpy()
    p1_np = pts1.detach().cpu().numpy()

    H_np, mask = cv2.findHomography(
        p1_np, 
        p0_np, 
        method=cv2.USAC_MAGSAC, 
        ransacReprojThreshold=5.0, 
        confidence=0.999, 
        maxIters=1000
    )

    if H_np is None:
        print("Homography estimation failed.")
        return None

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
    
    return stitched

def feature_detection_mapping(images):
    inputs = processor(images, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    image_sizes = [[(image.height, image.width) for image in images]]
    outputs = processor.post_process_keypoint_matching(outputs, image_sizes, threshold=0.2)
    return outputs

def apply_geometric_jitter(image, rotation_limit=10.0, translation_limit=5, perspective_limit=0.02):
    width, height = image.size
    
    angle = random.uniform(-rotation_limit, rotation_limit)
    tx = random.uniform(-translation_limit, translation_limit)
    ty = random.uniform(-translation_limit, translation_limit)

    img = image.rotate(angle, resample=Image.BILINEAR, translate=(tx, ty))

    coeffs = [
        1 + random.uniform(-perspective_limit, perspective_limit), 0, 0,
        0, 1 + random.uniform(-perspective_limit, perspective_limit), 0,
        random.uniform(-0.0001, 0.0001), random.uniform(-0.0001, 0.0001)
    ]
    
    return img.transform((width, height), Image.PERSPECTIVE, coeffs, Image.BILINEAR)

def apply_brightness_jitter(image, jitter_range=(0.7, 1.3)):
    enhancer = ImageEnhance.Brightness(image)
    factor = random.uniform(*jitter_range)
    return enhancer.enhance(factor)

def remove_alpha(img_rgba, bg_color=(0, 0, 0)):
    background = Image.new("RGB", img_rgba.size, bg_color)
    background.paste(img_rgba, mask=img_rgba.split()[3]) 
    return background

def split_image_variable_diagonal(image_path, min_overlap_pct=0.1):
    img = Image.open(image_path).convert("RGBA")
    w, h = img.size
    margin = 50 
    
    top_x = random.randint(margin, w - margin)
    
    max_slant = w
    min_overlap = int(w * min_overlap_pct)
    bottom_x = random.randint(max(margin, top_x - max_slant), 
                              min(w - margin, top_x + max_slant))
    
    mask_left = Image.new("L", (w, h), 0)
    draw_l = ImageDraw.Draw(mask_left)
    draw_l.polygon([(0, 0), (top_x + min_overlap, 0), (bottom_x + min_overlap, h), (0, h)], fill=255)
    
    mask_right = Image.new("L", (w, h), 0)
    draw_r = ImageDraw.Draw(mask_right)
    draw_r.polygon([(top_x - min_overlap, 0), (w, 0), (w, h), (bottom_x - min_overlap, h)], fill=255)

    left_img = img.copy()
    left_img.putalpha(mask_left)
    
    right_img = img.copy()
    right_img.putalpha(mask_right)

    cropped_left = remove_alpha(left_img)
    cropped_right = remove_alpha(right_img)

    return cropped_left, cropped_right, img

def process_images(files, rot_limit, trans_limit, persp_limit, bright_factor, overlap_pct):
    if files is None or len(files) == 0:
        return None, None, ""
    
    imgs = [Image.open(f.name) for f in files if True]
    if len(imgs) == 0:
        return None, None, ""
    
    def jitter_image(img):
        img = apply_geometric_jitter(img, rotation_limit=rot_limit, translation_limit=trans_limit, perspective_limit=persp_limit)
        img = apply_brightness_jitter(img, jitter_range=(max(0.1, bright_factor-0.3), min(2.0, bright_factor+0.3)))
        return img
    
    if len(imgs) == 1:
        img = imgs[0]
        original = img.copy()
        f_path = files[0].name
        
        left, right, _ = split_image_variable_diagonal(f_path, min_overlap_pct=overlap_pct)
        left = jitter_image(left)
        
        images_to_stitch = [left, right]
        feature_mapping = feature_detection_mapping(images_to_stitch)
        matches_viz = visualize_matches(images_to_stitch, feature_mapping)
        
        stitched = stitch_images(left, right, feature_mapping[0], device=device)
        if stitched is not None:
            result_img = Image.fromarray((stitched.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            ssim_score, _ = calculate_ssim(original, stitched)
            num_matches = (feature_mapping[0].get("matches0", torch.tensor([])) > -1).sum().item()
            return result_img, matches_viz, f"SSIM: {ssim_score:.4f}"
        else:
            jittered_original = jitter_image(img)
            return jittered_original, matches_viz, "Stitching failed"
    
    elif len(imgs) == 2:
        img1, img2 = imgs[0], imgs[1]
        feature_mapping = feature_detection_mapping([img1, img2])
        matches_viz = visualize_matches([img1, img2], feature_mapping)
        
        stitched = stitch_images(img1, img2, feature_mapping[0], device=device)
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
                trans_slider = gr.Slider(0, 20, value=3, step=0.5, label="Translation (px)")
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
        inputs=[input_files, rotation_slider, trans_slider, persp_slider, bright_slider, overlap_slider],
        outputs=[stitched_gallery, matches_gallery, stats_md]
    )

demo.launch(debug=True)
