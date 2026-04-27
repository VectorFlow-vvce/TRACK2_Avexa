"""
Falcon Integration Script
Loads trained segmentation model and runs inference on Falcon camera feed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2

# ============================================================================
# Model Architecture (must match training)
# ============================================================================

class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=7, padding=3),
            nn.GELU()
        )

        self.block = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.GELU(),
        )

        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block(x)
        return self.classifier(x)


# ============================================================================
# Segmentation Model Wrapper
# ============================================================================

class OffroadSegmentationModel:
    def __init__(self, model_path, backbone_size="small", device="cuda"):
        """
        Initialize the segmentation model for Falcon integration.
        
        Args:
            model_path: Path to segmentation_head.pth
            backbone_size: "small", "base", "large", or "giant"
            device: "cuda" or "cpu"
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Image dimensions (must match training)
        self.img_width = int(((960 / 2) // 14) * 14)
        self.img_height = int(((540 / 2) // 14) * 14)
        
        # Class names
        self.class_names = [
            "Background",
            "Trees",
            "Lush Bushes",
            "Dry Grass",
            "Dry Bushes",
            "Ground Clutter",
            "Logs",
            "Rocks",
            "Landscape",
            "Sky"
        ]
        
        # Color map for visualization
        self.color_map = np.array([
            [0, 0, 0],        # Background - Black
            [34, 139, 34],    # Trees - Forest Green
            [0, 255, 0],      # Lush Bushes - Bright Green
            [255, 255, 0],    # Dry Grass - Yellow
            [139, 69, 19],    # Dry Bushes - Brown
            [160, 82, 45],    # Ground Clutter - Sienna
            [101, 67, 33],    # Logs - Dark Brown
            [128, 128, 128],  # Rocks - Gray
            [210, 180, 140],  # Landscape - Tan
            [135, 206, 235]   # Sky - Sky Blue
        ], dtype=np.uint8)
        
        # Load backbone
        print("Loading DINOv2 backbone...")
        backbone_archs = {
            "small": "vits14",
            "base": "vitb14_reg",
            "large": "vitl14_reg",
            "giant": "vitg14_reg",
        }
        backbone_arch = backbone_archs[backbone_size]
        backbone_name = f"dinov2_{backbone_arch}"
        
        self.backbone = torch.hub.load(
            repo_or_dir="facebookresearch/dinov2",
            model=backbone_name
        )
        self.backbone.eval()
        self.backbone.to(self.device)
        print(f"✅ Loaded {backbone_name} backbone")
        
        # Get embedding dimension
        dummy_input = torch.randn(1, 3, self.img_height, self.img_width).to(self.device)
        with torch.no_grad():
            output = self.backbone.forward_features(dummy_input)["x_norm_patchtokens"]
        n_embedding = output.shape[2]
        
        # Load segmentation head
        print("Loading segmentation head...")
        self.classifier = SegmentationHeadConvNeXt(
            in_channels=n_embedding,
            out_channels=len(self.class_names),
            tokenW=self.img_width // 14,
            tokenH=self.img_height // 14
        )
        self.classifier.load_state_dict(torch.load(model_path, map_location=self.device))
        self.classifier.eval()
        self.classifier.to(self.device)
        print("✅ Model loaded successfully!")
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image, return_colored=True):
        """
        Run segmentation on an image.
        
        Args:
            image: PIL Image or numpy array (H, W, 3) in RGB
            return_colored: If True, return colored segmentation mask
            
        Returns:
            segmentation_mask: Colored mask if return_colored=True, else class IDs
            class_probabilities: Probability map for each class
        """
        # Convert to PIL if numpy
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        original_size = image.size  # (W, H)
        
        # Preprocess
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            # Get backbone features
            features = self.backbone.forward_features(img_tensor)["x_norm_patchtokens"]
            
            # Get segmentation logits
            logits = self.classifier(features)
            
            # Upsample to original image size
            logits = F.interpolate(
                logits,
                size=(original_size[1], original_size[0]),  # (H, W)
                mode="bilinear",
                align_corners=False
            )
            
            # Get class predictions
            class_probs = F.softmax(logits, dim=1)
            class_ids = torch.argmax(logits, dim=1)
        
        # Convert to numpy
        class_ids = class_ids.squeeze().cpu().numpy()
        class_probs = class_probs.squeeze().cpu().numpy()
        
        if return_colored:
            # Create colored mask
            colored_mask = self.color_map[class_ids]
            return colored_mask, class_probs
        else:
            return class_ids, class_probs
    
    def predict_from_falcon_camera(self, camera_image):
        """
        Process image from Falcon camera feed.
        
        Args:
            camera_image: Image from Falcon camera (numpy array or PIL Image)
            
        Returns:
            colored_mask: RGB segmentation mask
            overlay: Original image with mask overlay
        """
        # Convert to PIL if needed
        if isinstance(camera_image, np.ndarray):
            original_image = camera_image.copy()
            pil_image = Image.fromarray(camera_image)
        else:
            pil_image = camera_image
            original_image = np.array(pil_image)
        
        # Get segmentation
        colored_mask, probs = self.predict(pil_image, return_colored=True)
        
        # Create overlay (blend original + mask)
        overlay = cv2.addWeighted(original_image, 0.6, colored_mask, 0.4, 0)
        
        return colored_mask, overlay
    
    def get_navigable_area(self, image):
        """
        Identify navigable areas for robot navigation.
        
        Args:
            image: Input image
            
        Returns:
            navigable_mask: Binary mask where 1 = navigable, 0 = obstacle
            safe_direction: Recommended direction ("left", "center", "right", "stop")
        """
        class_ids, probs = self.predict(image, return_colored=False)
        
        # Define navigable classes (adjust based on your needs)
        navigable_classes = [0, 3, 5, 8]  # Background, Dry Grass, Ground Clutter, Landscape
        obstacle_classes = [1, 2, 4, 6, 7]  # Trees, Bushes, Logs, Rocks
        
        # Create navigable mask
        navigable_mask = np.isin(class_ids, navigable_classes).astype(np.uint8)
        
        # Analyze safe direction (divide image into left, center, right)
        h, w = navigable_mask.shape
        left_region = navigable_mask[:, :w//3]
        center_region = navigable_mask[:, w//3:2*w//3]
        right_region = navigable_mask[:, 2*w//3:]
        
        # Calculate navigability score for each region
        left_score = np.mean(left_region)
        center_score = np.mean(center_region)
        right_score = np.mean(right_region)
        
        # Determine safe direction
        if center_score > 0.6:
            safe_direction = "center"
        elif left_score > right_score and left_score > 0.5:
            safe_direction = "left"
        elif right_score > 0.5:
            safe_direction = "right"
        else:
            safe_direction = "stop"
        
        return navigable_mask, safe_direction


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Initialize model
    model = OffroadSegmentationModel(
        model_path="segmentation_head.pth",
        backbone_size="small",
        device="cuda"
    )
    
    # Test on a sample image
    test_image = Image.open("test_image.png")  # Replace with your test image
    
    # Get segmentation
    colored_mask, overlay = model.predict_from_falcon_camera(test_image)
    
    # Get navigation info
    navigable_mask, direction = model.get_navigable_area(test_image)
    
    print(f"Recommended direction: {direction}")
    
    # Save results
    cv2.imwrite("segmentation_result.png", cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
    cv2.imwrite("overlay_result.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print("✅ Results saved!")
