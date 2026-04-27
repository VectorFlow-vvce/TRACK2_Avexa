"""
Test trained model on unseen test images
For BigRock Exchange Hackathon
"""

import torch
from falcon_integration import OffroadSegmentationModel
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

# Value mapping from problem statement
value_map = {
    0: 0,        # background
    100: 1,      # Trees
    200: 2,      # Lush Bushes
    300: 3,      # Dry Grass
    500: 4,      # Dry Bushes
    550: 5,      # Ground Clutter
    600: 6,      # Flowers (if in dataset)
    700: 7,      # Logs
    800: 8,      # Rocks
    7100: 9,     # Landscape
    10000: 10    # Sky
}

def convert_mask(mask):
    """Convert raw mask values to class IDs"""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return new_arr

def compute_iou(pred, target, num_classes=10):
    """Compute IoU for each class"""
    iou_per_class = []
    
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id
        
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        
        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append(intersection / union)
    
    return iou_per_class, np.nanmean(iou_per_class)

def test_on_unseen_images(
    model_path="segmentation_head.pth",
    test_images_dir=None,
    test_masks_dir=None,
    output_dir="test_results"
):
    """
    Test model on unseen images
    
    Args:
        model_path: Path to trained model
        test_images_dir: Path to test images folder
        test_masks_dir: Path to test masks folder (if available)
        output_dir: Where to save results
    """
    print("=" * 60)
    print("Testing Model on Unseen Images")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print("\n1. Loading model...")
    model = OffroadSegmentationModel(
        model_path=model_path,
        backbone_size="small",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print("✅ Model loaded!")
    
    # Set default paths if not provided
    if test_images_dir is None:
        # Update this path to your actual test images location
        test_images_dir = r"C:\Users\Sanjana\Downloads\Offroad_Segmentation_testImages\Offroad_Segmentation_testImages\Color_Images"
    
    if test_masks_dir is None:
        # Check if segmentation masks exist for test set
        potential_mask_dir = r"C:\Users\Sanjana\Downloads\Offroad_Segmentation_testImages\Offroad_Segmentation_testImages\Segmentation"
        if os.path.exists(potential_mask_dir):
            test_masks_dir = potential_mask_dir
    
    if not os.path.exists(test_images_dir):
        print(f"❌ Test images directory not found: {test_images_dir}")
        print("Please update the path in the script")
        return
    
    # Get all test images
    test_images = [f for f in os.listdir(test_images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"\n2. Found {len(test_images)} test images")
    
    # Check if ground truth masks available
    has_ground_truth = test_masks_dir and os.path.exists(test_masks_dir)
    
    if has_ground_truth:
        print(f"✅ Ground truth masks found at: {test_masks_dir}")
    else:
        print("⚠️ No ground truth masks - will only generate predictions")
    
    # Process all test images
    print("\n3. Processing test images...")
    
    all_ious = []
    class_ious_sum = np.zeros(10)
    class_counts = np.zeros(10)
    
    predictions_dir = os.path.join(output_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    
    for img_name in tqdm(test_images, desc="Testing"):
        # Load image
        img_path = os.path.join(test_images_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        # Get prediction
        pred_mask_colored, _ = model.predict(image, return_colored=True)
        pred_mask, _ = model.predict(image, return_colored=False)
        
        # Save prediction
        pred_save_path = os.path.join(predictions_dir, f"pred_{img_name}")
        cv2.imwrite(pred_save_path, cv2.cvtColor(pred_mask_colored, cv2.COLOR_RGB2BGR))
        
        # Calculate IoU if ground truth available
        if has_ground_truth:
            mask_path = os.path.join(test_masks_dir, img_name)
            if os.path.exists(mask_path):
                gt_mask = Image.open(mask_path)
                gt_mask = convert_mask(gt_mask)
                
                class_ious, mean_iou = compute_iou(pred_mask, gt_mask, num_classes=10)
                all_ious.append(mean_iou)
                
                # Accumulate class-wise IoU
                for i, iou in enumerate(class_ious):
                    if not np.isnan(iou):
                        class_ious_sum[i] += iou
                        class_counts[i] += 1
    
    # Print results
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    if has_ground_truth and all_ious:
        mean_test_iou = np.mean(all_ious)
        print(f"\n📊 Overall Test IoU: {mean_test_iou:.4f} ({mean_test_iou*100:.2f}%)")
        
        # Class-wise IoU
        print("\n📊 Class-wise IoU:")
        class_names = [
            "Background", "Trees", "Lush Bushes", "Dry Grass", 
            "Dry Bushes", "Ground Clutter", "Logs", "Rocks", 
            "Landscape", "Sky"
        ]
        
        for i, name in enumerate(class_names):
            if class_counts[i] > 0:
                avg_iou = class_ious_sum[i] / class_counts[i]
                print(f"  {name:15s}: {avg_iou:.4f} ({avg_iou*100:.2f}%)")
        
        # Save results to file
        results_file = os.path.join(output_dir, "test_results.txt")
        with open(results_file, 'w') as f:
            f.write("TEST RESULTS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Overall Test IoU: {mean_test_iou:.4f} ({mean_test_iou*100:.2f}%)\n\n")
            f.write("Class-wise IoU:\n")
            for i, name in enumerate(class_names):
                if class_counts[i] > 0:
                    avg_iou = class_ious_sum[i] / class_counts[i]
                    f.write(f"  {name:15s}: {avg_iou:.4f} ({avg_iou*100:.2f}%)\n")
        
        print(f"\n✅ Results saved to: {results_file}")
    else:
        print("\n⚠️ No ground truth available - only predictions generated")
    
    print(f"\n✅ Predictions saved to: {predictions_dir}")
    print(f"✅ Processed {len(test_images)} test images")
    
    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)

if __name__ == "__main__":
    # Update these paths to match your dataset location
    test_on_unseen_images(
        model_path="segmentation_head.pth",
        test_images_dir=r"C:\Users\Sanjana\Downloads\Offroad_Segmentation_testImages\Offroad_Segmentation_testImages\Color_Images",
        test_masks_dir=r"C:\Users\Sanjana\Downloads\Offroad_Segmentation_testImages\Offroad_Segmentation_testImages\Segmentation",
        output_dir="test_results"
    )
