"""
Test your trained model before Falcon integration
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from falcon_integration import OffroadSegmentationModel

def test_model():
    print("=" * 60)
    print("Testing Trained Model")
    print("=" * 60)
    
    # Initialize model
    print("\n1. Loading model...")
    
    # Update this path to where your model file actually is
    # Option 1: If in same folder
    model_path = "segmentation_head.pth"
    
    # Option 2: If in models subfolder
    # model_path = "models/segmentation_head.pth"
    
    # Option 3: If in Downloads folder (update username)
    # model_path = r"C:\Users\Sanjana\Downloads\segmentation_head.pth"
    
    model = OffroadSegmentationModel(
        model_path=model_path,
        backbone_size="small",  # Change if you used different size
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print("✅ Model loaded successfully!")
    
    # Load a test image (use one from your validation set)
    print("\n2. Loading test image...")
    
    # Using your actual dataset path
    test_image_path = r"C:\Users\Sanjana\Downloads\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\val\Color_Images\ww10000592.png"
    
    try:
        test_image = Image.open(test_image_path)
        print(f"✅ Loaded image: {test_image.size}")
    except Exception as e:
        print(f"⚠️ Could not load image from: {test_image_path}")
        print(f"Error: {e}")
        print("\n💡 Please update the test_image_path in test_model.py")
        print("   Point it to an image from your validation dataset")
        return
    
    # Run segmentation
    print("\n3. Running segmentation...")
    colored_mask, overlay = model.predict_from_falcon_camera(test_image)
    print("✅ Segmentation complete!")
    
    # Get navigation info
    print("\n4. Analyzing navigation...")
    navigable_mask, direction = model.get_navigable_area(test_image)
    print(f"✅ Recommended direction: {direction}")
    
    # Visualize results
    print("\n5. Visualizing results...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(test_image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(colored_mask)
    axes[0, 1].set_title("Segmentation Mask")
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title("Overlay")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(navigable_mask, cmap='gray')
    axes[1, 1].set_title(f"Navigable Areas\nDirection: {direction}")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig("test_results.png", dpi=150, bbox_inches='tight')
    print("✅ Results saved to 'test_results.png'")
    plt.show()
    
    print("\n" + "=" * 60)
    print("✅ Model test complete! Ready for Falcon integration.")
    print("=" * 60)

if __name__ == "__main__":
    test_model()
