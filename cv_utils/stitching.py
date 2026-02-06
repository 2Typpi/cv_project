import torch
import kornia
import cv2
import torchvision.transforms as T

def feature_detection_mapping(images, processor, model, device):
    inputs = processor(images, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    image_sizes = [[(image.height, image.width) for image in images]]
    outputs = processor.post_process_keypoint_matching(outputs, image_sizes, threshold=0.2)
    return outputs

def stitch_images(img0_pil, img1_pil, output, device):
    to_tensor = T.ToTensor()
    image0 = to_tensor(img0_pil).to(device)
    image1 = to_tensor(img1_pil).to(device)

    pts0 = output["keypoints0"].float()
    pts1 = output["keypoints1"].float()

    num_matches = pts0.shape[0]
    
    if num_matches < 4:
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
        return None, num_matches

    H = torch.from_numpy(H_np).to(device).float()

    _, h0, w0 = image0.shape
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