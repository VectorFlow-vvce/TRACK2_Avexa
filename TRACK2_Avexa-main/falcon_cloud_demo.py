"""
FalconCloud Integration Demo
Connects your segmentation model to FalconCloud simulation
"""

from falcon_integration import OffroadSegmentationModel
import numpy as np
import cv2
import time
import json

# TODO: Import FalconCloud SDK (adjust based on actual SDK)
# from falcon_cloud import FalconCloudClient, Simulation
# OR
# from duality import FalconCloud

class FalconCloudSegmentationDemo:
    def __init__(self, model_path="segmentation_head.pth", api_key=None):
        """
        Initialize FalconCloud demo with your trained model.
        
        Args:
            model_path: Path to your trained model
            api_key: Your FalconCloud API key
        """
        print("=" * 60)
        print("FalconCloud Segmentation Demo")
        print("=" * 60)
        
        # Load segmentation model
        print("\n1. Loading segmentation model...")
        self.model = OffroadSegmentationModel(
            model_path=model_path,
            backbone_size="small",
            device="cpu"  # Using CPU since we're on local machine
        )
        print("✅ Model loaded!")
        
        # Initialize FalconCloud connection
        print("\n2. Connecting to FalconCloud...")
        self.api_key = api_key
        
        # TODO: Initialize FalconCloud client (adjust based on actual SDK)
        # self.falcon = FalconCloudClient(api_key=api_key)
        # self.simulation = None
        # self.camera = None
        
        print("⚠️ FalconCloud SDK not configured yet")
        print("   Please check Duality AI documentation for SDK setup")
        
        # Stats tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.directions_history = []
    
    def connect_to_simulation(self, simulation_id=None, environment_name=None):
        """
        Connect to a FalconCloud simulation.
        
        Args:
            simulation_id: ID of existing simulation
            environment_name: Name of environment to load (e.g., "offroad_scene")
        """
        print("\n3. Connecting to simulation...")
        
        # TODO: Connect to FalconCloud simulation (adjust based on actual SDK)
        # Option 1: Connect to existing simulation
        # if simulation_id:
        #     self.simulation = self.falcon.get_simulation(simulation_id)
        # 
        # Option 2: Create new simulation
        # else:
        #     self.simulation = self.falcon.create_simulation(
        #         environment=environment_name,
        #         config={
        #             "camera": {"resolution": [960, 540]},
        #             "robot": {"type": "ground_vehicle"}
        #         }
        #     )
        # 
        # self.camera = self.simulation.get_camera("main_camera")
        # self.robot = self.simulation.get_robot()
        
        print("⚠️ Simulation connection not implemented")
        print("   Waiting for FalconCloud SDK details...")
    
    def process_frame(self, frame):
        """
        Process a single frame from FalconCloud camera.
        
        Args:
            frame: Image from FalconCloud (numpy array, RGB)
            
        Returns:
            overlay: Segmentation overlay
            direction: Navigation direction
            metrics: Performance metrics
        """
        # Run segmentation
        colored_mask, overlay = self.model.predict_from_falcon_camera(frame)
        
        # Get navigation recommendation
        navigable_mask, direction = self.model.get_navigable_area(frame)
        
        # Track statistics
        self.frame_count += 1
        self.directions_history.append(direction)
        
        # Calculate metrics
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        metrics = {
            "frame": self.frame_count,
            "direction": direction,
            "fps": fps,
            "navigable_percentage": np.mean(navigable_mask) * 100
        }
        
        return overlay, direction, metrics
    
    def run_demo_local_test(self, video_path=None):
        """
        Test demo locally with video file or webcam before FalconCloud.
        
        Args:
            video_path: Path to test video, or None for webcam
        """
        print("\n" + "=" * 60)
        print("Running LOCAL TEST (not FalconCloud yet)")
        print("Press 'q' to quit")
        print("=" * 60)
        
        # Open video source
        if video_path:
            cap = cv2.VideoCapture(video_path)
            print(f"Testing with video: {video_path}")
        else:
            cap = cv2.VideoCapture(0)
            print("Testing with webcam")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            overlay, direction, metrics = self.process_frame(frame_rgb)
            
            # Convert back to BGR for display
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            
            # Add info overlay
            info_text = [
                f"Frame: {metrics['frame']}",
                f"Direction: {metrics['direction']}",
                f"FPS: {metrics['fps']:.1f}",
                f"Navigable: {metrics['navigable_percentage']:.1f}%"
            ]
            
            y_offset = 30
            for text in info_text:
                cv2.putText(overlay_bgr, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 35
            
            # Display
            cv2.imshow("FalconCloud Demo (Local Test)", overlay_bgr)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print summary
        print("\n" + "=" * 60)
        print("Demo Summary")
        print("=" * 60)
        print(f"Total frames: {self.frame_count}")
        print(f"Average FPS: {metrics['fps']:.1f}")
        print(f"Direction distribution:")
        from collections import Counter
        direction_counts = Counter(self.directions_history)
        for direction, count in direction_counts.items():
            percentage = (count / len(self.directions_history)) * 100
            print(f"  {direction}: {count} ({percentage:.1f}%)")
    
    def run_falcon_cloud_demo(self):
        """
        Run demo with actual FalconCloud connection.
        """
        print("\n" + "=" * 60)
        print("Running FalconCloud Demo")
        print("=" * 60)
        
        # TODO: Implement FalconCloud loop (adjust based on actual SDK)
        # while True:
        #     # Get frame from FalconCloud
        #     frame = self.camera.capture()
        #     
        #     # Process frame
        #     overlay, direction, metrics = self.process_frame(frame)
        #     
        #     # Send navigation command to robot
        #     self.robot.set_direction(direction)
        #     
        #     # Optional: Send overlay back to FalconCloud for visualization
        #     self.simulation.display_overlay(overlay)
        #     
        #     # Log metrics
        #     print(f"Frame {metrics['frame']}: {direction} | FPS: {metrics['fps']:.1f}")
        #     
        #     # Check for stop condition
        #     if self.simulation.is_complete():
        #         break
        
        print("⚠️ FalconCloud demo loop not implemented")
        print("   Waiting for SDK documentation...")
    
    def save_demo_video(self, output_path="demo_output.mp4"):
        """
        Save demo video with segmentation overlay.
        """
        # TODO: Implement video recording
        pass


def main():
    """
    Main entry point for FalconCloud demo.
    """
    # Configuration
    MODEL_PATH = "segmentation_head.pth"
    FALCON_API_KEY = "your_api_key_here"  # Get from Duality AI dashboard
    
    # Initialize demo
    demo = FalconCloudSegmentationDemo(
        model_path=MODEL_PATH,
        api_key=FALCON_API_KEY
    )
    
    # Option 1: Test locally first (recommended)
    print("\n" + "=" * 60)
    print("STEP 1: Local Testing")
    print("=" * 60)
    print("\nTesting model locally before FalconCloud...")
    print("This helps verify everything works!")
    
    # Test with webcam
    demo.run_demo_local_test()
    
    # Option 2: Connect to FalconCloud (after local test works)
    # print("\n" + "=" * 60)
    # print("STEP 2: FalconCloud Connection")
    # print("=" * 60)
    # demo.connect_to_simulation(environment_name="offroad_environment")
    # demo.run_falcon_cloud_demo()


if __name__ == "__main__":
    main()
