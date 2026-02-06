import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

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

    # ensure the data types and ranges match
    orig_np = orig_np.astype(np.float32) / 255.0 if orig_np.max() > 1.0 else orig_np.astype(np.float32)
    stitch_np = stitch_np.astype(np.float32) / 255.0 if stitch_np.max() > 1.0 else stitch_np.astype(np.float32)
    
    # conversion to gray scale
    orig_np = np.dot(orig_np[..., :3], [0.2989, 0.5870, 0.1140]) if gray_scale else orig_np
    stitch_np = np.dot(stitch_np[..., :3], [0.2989, 0.5870, 0.1140]) if gray_scale else stitch_np
    
    score, diff = ssim(orig_np, stitch_np, full=True, data_range=1.0, channel_axis=None if gray_scale else 2)

    return score, diff