import torch
from transformers import AutoImageProcessor, AutoModel
from .stitching import feature_detection_mapping, stitch_images
from .metrics import calculate_ssim

class Stitcher:
    def __init__(self, model_name="ETH-CVG/lightglue_superpoint"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)

    def stitch(self, img0_pil, img1_pil):
        feature_mapping = feature_detection_mapping([img0_pil, img1_pil], self.processor, self.model, self.device)
        stitched_image, num_matches = stitch_images(img0_pil, img1_pil, feature_mapping[0], self.device)
        return stitched_image, feature_mapping, num_matches

    def visualize_matches(self, images, feature_mapping):
        """Visualization """
        if len(feature_mapping) == 0:
            return images[0]
        
        matches_viz = self.processor.visualize_keypoint_matching(images, feature_mapping)
        return matches_viz[0]