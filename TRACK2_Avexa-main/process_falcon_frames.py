"""
Batch process Falcon frames
"""

from falcon_integration import OffroadSegmentationModel
from PIL import Image
import cv2
import os
import numpy as np
from tqdm import tqdm

def process_all_falcon_frames(
    frames_dir="falcon_frames",
    output_dir="falcon_results",
    model_path="segmentation_head.pth"
):
    """
    Process all Falcon frames at once
    """
    print("=" * 60)
    print("Processing Falcon Frames")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "segmentation"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "overlay"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "combined"), exist_ok=True)
    
    # Load model
    print("\n1. Loading model...")
    model = OffroadSegmentationModel(
        model_path=model_path,
        backbone_size="small"
    )
    print("✅ Model loaded!")
    
    # Get all frame files
    if not os.path.exists(frames_dir):
        print(f"\n❌ Frames directory not found: {frames_dir}")
        print("Please create it and add your Falcon screenshots")
        return
    
    frame_files = [f for f in os.listdir(frames_dir) 
                   if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(frame_files) == 0:
        print(f"\n❌ No images found in {frames_dir}")
        print("Please add Falcon screenshots to this folder")
        return
    
    print(f"\n2. Found {len(frame_files)} frames")
    
    # Process each frame
    print("\n3. Processing frames...")
    
    results = []
    
    for frame_file in tqdm(frame_files, desc="Processing"):
        # Load frame
        frame_path = os.path.join(frames_dir, frame_file)
        image = Image.open(frame_path).convert("RGB")
        
        # Run segmentation
        colored_mask, overlay = model.predict_from_falcon_camera(image)
        navigable_mask, direction = model.get_navigable_area(image)
        
        # Save individual results
        base_name = os.path.splitext(frame_file)[0]
        
        # Save segmentation
        seg_path = os.path.join(output_dir, "segmentation", f"{base_name}_seg.png")
        cv2.imwrite(seg_path, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
        
        # Save overlay
        overlay_path = os.path.join(output_dir, "overlay", f"{base_name}_overlay.png")
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        # Create combined view (original + segmentation + overlay)
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image)
        axes[0].set_title("Falcon Scene", fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(colored_mask)
        axes[1].set_title("Segmentation", fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay\nDirection: {direction}", 
                         fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        combined_path = os.path.join(output_dir, "combined", f"{base_name}_combined.png")
        plt.savefig(combined_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        results.append({
            'frame': frame_file,
            'direction': direction,
            'navigable_percent': np.mean(navigable_mask) * 100
        })
    
    # Save summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Falcon Frames Processing Summary\n")
        f.write("=" * 60 + "\n\n")
        
        for result in results:
            line = f"{result['frame']:30s} | Direction: {result['direction']:6s} | Navigable: {result['navigable_percent']:.1f}%"
            print(line)
            f.write(line + "\n")
    
    print("\n" + "=" * 60)
    print("✅ Processing complete!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - Segmentation masks: {output_dir}/segmentation/")
    print(f"  - Overlays: {output_dir}/overlay/")
    print(f"  - Combined views: {output_dir}/combined/")
    print(f"  - Summary: {output_dir}/summary.txt")
    
    print("\n💡 Use the 'combined' images for your presentation!")

if __name__ == "__main__":
    # Create falcon_frames directory if it doesn't exist
    if not os.path.exists("falcon_frames"):
        os.makedirs("falcon_frames")
        print("📁 Created 'falcon_frames' directory")
        print("Please add your Falcon screenshots to this folder")
        print("Then run this script again")
    else:
        process_all_falcon_frames()
