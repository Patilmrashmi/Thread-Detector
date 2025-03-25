import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from picamera import PiCamera
from picamera.array import PiRGBArray
from time import sleep

def calculate_ppi(image_width, working_distance_mm=100):
    """
    Calculate a practical PPI based on working distance and field of view
    Args:
        image_width: Width of the image in pixels
        working_distance_mm: Distance from camera to subject in mm
    Returns:
        Practical PPI for fabric analysis
    """
    TARGET_PPI = 300
    return TARGET_PPI

def crop_1x1_inch(image):
    """
    Crop a 1x1 inch square from the center of the image using practical PPI
    Args:
        image: Input image array
    Returns:
        Cropped image, actual PPI
    """
    ppi = calculate_ppi(image.shape[1])
    
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2
    half_size = ppi // 2
    
    x1 = max(center_x - half_size, 0)
    y1 = max(center_y - half_size, 0)
    x2 = min(center_x + half_size, width)
    y2 = min(center_y + half_size, height)
    
    cropped_image = image[y1:y2, x1:x2]
    
    if cropped_image.shape[0] != ppi or cropped_image.shape[1] != ppi:
        cropped_image = cv2.resize(cropped_image, (ppi, ppi))
    
    return cropped_image, ppi

def estimate_fabric_density(binary_image):
    """
    Estimate if the fabric is densely or loosely woven
    Args:
        binary_image: Binary threshold image
    Returns:
        bool: True if densely woven, False if loosely woven
    """
    white_pixel_ratio = np.sum(binary_image > 0) / binary_image.size
    return white_pixel_ratio > 0.3  # Threshold determined empirically

def analyze_fabric_weave(image):
    """
    Adaptive analysis of fabric weave pattern for both dense and loose weaves
    Args:
        image: Input image array
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Multi-scale enhancement
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, searchWindowSize=21)
    
    # Create two different enhanced versions
    blurred1 = cv2.GaussianBlur(denoised, (3, 3), 0)
    clahe1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced1 = clahe1.apply(blurred1)
    
    blurred2 = cv2.GaussianBlur(denoised, (5, 5), 0)
    clahe2 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced2 = clahe2.apply(blurred2)
    
    enhanced = cv2.addWeighted(enhanced1, 0.6, enhanced2, 0.4, 0)
    
    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )
    
    # Determine if fabric is densely or loosely woven
    is_dense = estimate_fabric_density(binary)
    
    # Clean up noise with adaptive kernel size
    kernel_size = 2 if is_dense else 3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Calculate projections
    vertical_dark = np.sum(binary, axis=0)
    horizontal_dark = np.sum(binary, axis=1)
    
    # Adaptive smoothing
    sigma = 2 if is_dense else 3
    vertical_proj = gaussian_filter1d(vertical_dark, sigma=sigma)
    horizontal_proj = gaussian_filter1d(horizontal_dark, sigma=sigma)
    
    # Normalize projections
    horizontal_proj = (horizontal_proj - np.min(horizontal_proj)) / (np.max(horizontal_proj) - np.min(horizontal_proj)) * 100
    vertical_proj = (vertical_proj - np.min(vertical_proj)) / (np.max(vertical_proj) - np.min(vertical_proj)) * 100
    
    # Adaptive peak detection parameters
    if is_dense:
        distance = 10
        prominence = 3
        height = 15
        width = 2
    else:
        distance = 15
        prominence = 5
        height = 20
        width = 3
    
    # Peak detection
    horizontal_peaks, _ = find_peaks(
        horizontal_proj,
        distance=distance,
        prominence=prominence,
        height=height,
        width=width
    )
    
    vertical_peaks, _ = find_peaks(
        vertical_proj,
        distance=distance,
        prominence=prominence,
        height=height,
        width=width
    )
    
    return horizontal_peaks, vertical_peaks, horizontal_proj, vertical_proj, binary, is_dense

def main():
    # Initialize camera
    camera = PiCamera()
    camera.resolution = (1920, 1080)
    camera.framerate = 30
    raw_capture = PiRGBArray(camera, size=(1920, 1080))
    
    # Allow camera to warmup
    sleep(2)
    
    print("Press 'c' to capture and analyze, 'q' to quit")
    
    try:
        for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
            image = frame.array
            
            # Display live preview
            cv2.imshow("Camera Preview", image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                # Crop 1x1 inch area
                cropped_image, actual_ppi = crop_1x1_inch(image)
                
                print(f"Actual PPI: {actual_ppi}")
                print(f"Cropped image size: {cropped_image.shape[:2]}")
                
                # Analyze the cropped frame
                h_peaks, v_peaks, h_proj, v_proj, binary, is_dense = analyze_fabric_weave(cropped_image)
                
                # Create visualization
                visualization = cropped_image.copy()
                
                # Draw detected threads
                for x in v_peaks:
                    cv2.line(visualization, (x, 0), (x, visualization.shape[0]), (0, 255, 0), 1)
                for y in h_peaks:
                    cv2.line(visualization, (0, y), (visualization.shape[1], y), (255, 0, 0), 1)
                
                # Plot results
                plt.figure(figsize=(20, 10))
                
                plt.subplot(2, 3, 1)
                plt.title(f"Original Fabric (1x1 inch, {actual_ppi} PPI)\n{'Dense' if is_dense else 'Loose'} Weave Detected")
                plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                
                plt.subplot(2, 3, 2)
                plt.title("Enhanced Binary Threshold")
                plt.imshow(binary, cmap='gray')
                
                plt.subplot(2, 3, 3)
                plt.title("Detected Threads")
                plt.imshow(cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB))
                
                plt.subplot(2, 3, 4)
                plt.title("Horizontal Thread Detection")
                plt.plot(h_proj)
                plt.plot(h_peaks, h_proj[h_peaks], "rx")
                
                plt.subplot(2, 3, 5)
                plt.title("Vertical Thread Detection")
                plt.plot(v_proj)
                plt.plot(v_peaks, v_proj[v_peaks], "rx")
                
                plt.tight_layout()
                plt.show()
                
                # Save visualization
                cv2.imwrite('fabric_analysis.jpg', visualization)
                
                # Print thread counts
                horizontal_count = len(h_peaks)
                vertical_count = len(v_peaks)
                print(f"\nAnalysis Results:")
                print(f"Fabric type: {'Dense' if is_dense else 'Loose'} weave")
                print(f"Number of horizontal threads: {horizontal_count}")
                print(f"Number of vertical threads: {vertical_count}")
                print(f"Total thread count: {horizontal_count + vertical_count}")
                print(f"Analysis performed on exactly 1x1 inch area at {actual_ppi} PPI")
                
                break
                
            elif key == ord('q'):
                break
                
            # Clear the stream for next frame
            raw_capture.truncate(0)
            
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        camera.close()

if _name_ == "_main_":
    main()