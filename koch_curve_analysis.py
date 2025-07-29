import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import measure, filters, morphology
from skimage.color import rgb2gray
import os

class KochCurveAnalyzer:
    def __init__(self, image_path):
        """
        Initialize with Koch curve image
        """
        self.image_path = image_path
        self.image = None
        self.contour_points = None
        
    def load_and_preprocess_image(self):
        """
        Load image and preprocess for contour extraction
        Handles small images (< 200x200) with appropriate scaling
        """
        # Load image
        self.image = plt.imread(self.image_path)
        
        # Handle different image formats (RGB, RGBA, Grayscale)
        if len(self.image.shape) == 3:
            height, width, channels = self.image.shape
            print(f"Image format: {width}x{height} with {channels} channels")
            
            if channels == 4:  # RGBA format
                # Remove alpha channel or use it for transparency
                if self.image[:, :, 3].min() < 1.0:  # Has transparency
                    # Composite with white background
                    alpha = self.image[:, :, 3:4]
                    rgb = self.image[:, :, :3]
                    self.image = alpha * rgb + (1 - alpha) * 1.0  # White background
                else:
                    # No transparency, just remove alpha channel
                    self.image = self.image[:, :, :3]
                    
            elif channels == 3:  # RGB format
                pass  # No change needed
            else:
                raise ValueError(f"Unexpected number of channels: {channels}")
                
            # Convert to grayscale
            gray = rgb2gray(self.image)
            
        elif len(self.image.shape) == 2:
            # Already grayscale
            height, width = self.image.shape
            gray = self.image
            print(f"Image format: {width}x{height} grayscale")
        else:
            raise ValueError(f"Unexpected image shape: {self.image.shape}")
        
        print(f"Original image size: {width}x{height}")
        
        # Scale up small images for better contour detection
        if width < 200 or height < 200:
            scale_factor = max(200/width, 200/height)
            # Use nearest neighbor to preserve sharp edges
            from skimage.transform import rescale
            gray = rescale(gray, scale_factor, anti_aliasing=False, preserve_range=True)
            print(f"Scaled image by factor {scale_factor:.2f} to: {gray.shape[1]}x{gray.shape[0]}")
            
        # Ensure values are in [0,1] range
        if gray.max() > 1:
            gray = gray / 255.0
            
        # Invert if needed (make curve white on black background)
        if np.mean(gray) > 0.5:  # If background is light
            gray = 1 - gray
            
        # Apply threshold to get binary image
        # Use a more sensitive threshold for small/simple images
        try:
            threshold = filters.threshold_otsu(gray)
        except ValueError:
            # If Otsu fails (e.g., very simple image), use mean as threshold
            threshold = np.mean(gray)
            
        binary = gray > threshold
        
        # Adjust cleaning parameters for small images
        min_object_size = max(5, int(0.001 * binary.size))  # Scale with image size
        disk_size = 1 if min(binary.shape) < 100 else 2
        
        # Clean up the binary image
        binary = morphology.remove_small_objects(binary, min_size=min_object_size)
        binary = morphology.closing(binary, morphology.disk(disk_size))
        
        return binary
    
    def extract_contour_from_image(self):
        """
        Extract contour points from Koch curve image using edge detection
        Optimized for small images with adaptive parameters
        """
        binary = self.load_and_preprocess_image()
        
        # For very small images, try multiple contour detection methods
        contours = []
        
        # Method 1: Standard contour detection
        try:
            contours_method1 = measure.find_contours(binary, level=0.5)
            contours.extend(contours_method1)
        except Exception as e:
            print(f"Standard contour detection failed: {e}")
        
        # Method 2: Edge detection for small/simple images
        if not contours or all(len(c) < 10 for c in contours):
            from skimage.feature import canny
            from skimage.measure import label, regionprops
            
            # Apply Canny edge detection with adjusted parameters for small images
            sigma = 0.5 if min(binary.shape) < 100 else 1.0
            edges = canny(binary.astype(float), sigma=sigma, low_threshold=0.1, high_threshold=0.2)
            
            try:
                edge_contours = measure.find_contours(edges, level=0.5)
                contours.extend(edge_contours)
                print("Used edge detection for contour extraction")
            except Exception as e:
                print(f"Edge detection method failed: {e}")
        
        # Method 3: Extract boundary of connected components
        if not contours or all(len(c) < 10 for c in contours):
            from skimage.segmentation import find_boundaries
            boundaries = find_boundaries(binary, mode='thick')
            
            try:
                boundary_contours = measure.find_contours(boundaries, level=0.5)
                contours.extend(boundary_contours)
                print("Used boundary detection for contour extraction")
            except Exception as e:
                print(f"Boundary detection method failed: {e}")
        
        # Select the longest contour
        if not contours:
            raise ValueError("No contours found in image with any method")
        
        # Filter out very short contours (likely noise)
        min_length = max(10, int(0.01 * max(binary.shape)))
        valid_contours = [c for c in contours if len(c) >= min_length]
        
        if not valid_contours:
            # If no long contours, use the longest available
            valid_contours = contours
            
        longest_contour = max(valid_contours, key=len)
        
        # Store contour points (skimage returns (row, col) which corresponds to (y, x))
        # Keep as (row, col) = (y, x) to maintain proper orientation
        self.contour_points = np.column_stack([longest_contour[:, 0], longest_contour[:, 1]])
        
        print(f"Extracted contour with {len(self.contour_points)} points")
        print(f"Image dimensions: {binary.shape}")
        print(f"Contour density: {len(self.contour_points) / (binary.shape[0] * binary.shape[1]):.6f} points/pixel")
        
        return self.contour_points
    
    def visualize_contour(self, title_suffix="", save_path=None, figsize=(12, 8), dpi=300):
        """
        Visualize the extracted contour
        """
        if self.contour_points is None:
            self.extract_contour_from_image()
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Original image
        ax1.imshow(self.image, cmap='gray')
        ax1.set_title('Original Koch Curve Image')
        ax1.axis('off')
        
        # Extracted contour
        ax2.plot(self.contour_points[:, 1], self.contour_points[:, 0], 'b-', linewidth=1, alpha=0.7)
        ax2.scatter(self.contour_points[:, 1], self.contour_points[:, 0], c='red', s=0.5, alpha=0.5)
        ax2.set_xlabel('X coordinate')
        ax2.set_ylabel('Y coordinate')
        ax2.set_title(f'Extracted Contour{title_suffix}')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        # Invert Y axis to match image coordinates
        ax2.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = 'koch_curve_contour_extraction.png'
            
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Contour visualization saved as: {save_path}")
        
        plt.show()
        return self.contour_points, save_path

def box_counting_fractal_dimension(contour_points, box_sizes=None, save_path=None, 
                                  figsize=(10, 6), dpi=300):
    """
    Calculate fractal dimension using box-counting method
    Optimized for small images with adaptive box sizes
    """
    # Normalize contour to [0,1] × [0,1] unit square
    x_min, x_max = contour_points[:, 0].min(), contour_points[:, 0].max()
    y_min, y_max = contour_points[:, 1].min(), contour_points[:, 1].max()
    
    # Avoid division by zero
    x_range = x_max - x_min if x_max != x_min else 1.0
    y_range = y_max - y_min if y_max != y_min else 1.0
    
    # Normalize to [0,1] range
    x_norm = (contour_points[:, 0] - x_min) / x_range
    y_norm = (contour_points[:, 1] - y_min) / y_range
    normalized_points = np.column_stack([x_norm, y_norm])
    
    # Adaptive box sizes based on number of contour points
    if box_sizes is None:
        num_points = len(contour_points)
        
        if num_points < 100:
            # For very small contours, use larger minimum box size
            min_box_size = 0.05
            max_box_size = 0.8
            num_sizes = 10
        elif num_points < 500:
            # For small contours
            min_box_size = 0.02
            max_box_size = 0.6
            num_sizes = 15
        else:
            # For larger contours (original parameters)
            min_box_size = 0.01
            max_box_size = 0.5
            num_sizes = 20
            
        box_sizes = np.logspace(np.log10(min_box_size), np.log10(max_box_size), num_sizes)
        print(f"Using {num_sizes} box sizes from {min_box_size:.3f} to {max_box_size:.3f}")
    
    counts = []
    
    for epsilon in box_sizes:
        occupied_boxes = set()
        
        for point in normalized_points:
            box_x = int(point[0] / epsilon)
            box_y = int(point[1] / epsilon)
            occupied_boxes.add((box_x, box_y))
        
        counts.append(len(occupied_boxes))
    
    # Filter out box sizes that result in too few boxes (less reliable)
    min_boxes = 3
    valid_indices = [i for i, count in enumerate(counts) if count >= min_boxes]
    
    if len(valid_indices) < 3:
        print("Warning: Very few valid box sizes for reliable fractal dimension calculation")
        valid_indices = range(len(counts))  # Use all data points
    
    valid_box_sizes = box_sizes[valid_indices]
    valid_counts = np.array(counts)[valid_indices]
    
    # Fit line to log-log plot
    log_sizes = np.log(1/valid_box_sizes)
    log_counts = np.log(valid_counts)
    
    # Linear regression
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    fractal_dimension = coeffs[0]
    
    # Calculate R-squared for goodness of fit
    y_pred = coeffs[0] * log_sizes + coeffs[1]
    ss_res = np.sum((log_counts - y_pred) ** 2)
    ss_tot = np.sum((log_counts - np.mean(log_counts)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Plot results
    plt.figure(figsize=figsize)
    plt.loglog(1/box_sizes, counts, 'bo-', label='All Data', markersize=4, alpha=0.7)
    plt.loglog(1/valid_box_sizes, valid_counts, 'ro-', label='Used for Fit', markersize=6)
    plt.loglog(1/valid_box_sizes, np.exp(coeffs[1]) * (1/valid_box_sizes)**coeffs[0], 
               'r--', label=f'Fit: D = {fractal_dimension:.3f} (R² = {r_squared:.3f})', linewidth=2)
    plt.xlabel('1/Box Size (1/ε)')
    plt.ylabel('Number of Occupied Boxes N(ε)')
    plt.title('Box-Counting Fractal Dimension Analysis\nKoch Curve (Optimized for Small Images)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path is None:
        save_path = 'koch_curve_fractal_analysis.png'
    
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    print(f"Fractal analysis plot saved as: {save_path}")
    print(f"Fractal dimension: {fractal_dimension:.4f}")
    print(f"R-squared (goodness of fit): {r_squared:.4f}")
    print(f"Used {len(valid_indices)} out of {len(box_sizes)} box sizes for calculation")
    
    plt.show()
    
    return fractal_dimension, box_sizes, counts, save_path

def analyze_koch_curve_image(image_path):
    """
    Complete analysis of a Koch curve image
    """
    print(f"Analyzing Koch curve image: {image_path}")
    
    # Create analyzer
    analyzer = KochCurveAnalyzer(image_path)
    
    # Extract contour
    contour_points = analyzer.extract_contour_from_image()
    
    # Visualize contour extraction
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    contour_save_path = f"{base_name}_contour_extraction.png"
    analyzer.visualize_contour(f" - {base_name}", save_path=contour_save_path)
    
    # Calculate fractal dimension
    fractal_save_path = f"{base_name}_fractal_analysis.png"
    fractal_dim, box_sizes, counts, _ = box_counting_fractal_dimension(
        contour_points, save_path=fractal_save_path)
    
    print(f"\nResults for {base_name}:")
    print(f"- Contour points: {len(contour_points)}")
    print(f"- Fractal dimension: {fractal_dim:.4f}")
    print(f"- Theoretical Koch curve dimension: 1.2618 (log(4)/log(3))")
    print(f"- Difference from theoretical: {abs(fractal_dim - 1.2618):.4f}")
    
    return {
        'image_path': image_path,
        'contour_points': len(contour_points),
        'fractal_dimension': fractal_dim,
        'theoretical_dimension': 1.2618,
        'difference': abs(fractal_dim - 1.2618)
    }

# Example usage
if __name__ == "__main__":
    # Replace with your Koch curve image path
    image_path = "1.1.png"  # Change this to your image file
    
    try:
        results = analyze_koch_curve_image(image_path)
        print("\nAnalysis complete!")
        
    except FileNotFoundError:
        print(f"Image file not found: {image_path}")
        print("Please update the 'image_path' variable with the correct path to your Koch curve image.")
        print("\nFor small images (< 200x200 pixels):")
        print("- The script will automatically scale up the image for better contour detection")
        print("- Uses adaptive parameters optimized for small image analysis")
        print("- Multiple contour detection methods ensure robust extraction")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Make sure you have the required libraries installed:")
        print("pip install opencv-python scikit-image matplotlib numpy")
        print("\nFor small images, this script includes:")
        print("- Automatic image scaling (preserves edge sharpness)")
        print("- Adaptive thresholding")
        print("- Multiple contour detection methods")
        print("- Optimized box-counting parameters")

# For analyzing multiple Koch curve images
def analyze_multiple_koch_curves(image_folder):
    """
    Analyze multiple Koch curve images in a folder
    """
    results = []
    
    # Supported image formats
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(supported_formats):
            image_path = os.path.join(image_folder, filename)
            try:
                result = analyze_koch_curve_image(image_path)
                results.append(result)
            except Exception as e:
                print(f"Error analyzing {filename}: {e}")
                continue
    
    # Summary
    if results:
        print(f"\n{'='*60}")
        print("SUMMARY OF ALL KOCH CURVE ANALYSES")
        print(f"{'='*60}")
        
        for result in results:
            name = os.path.basename(result['image_path'])
            print(f"{name:30} | D = {result['fractal_dimension']:.4f} | "
                  f"Error = {result['difference']:.4f}")
        
        avg_dimension = np.mean([r['fractal_dimension'] for r in results])
        print(f"\nAverage fractal dimension: {avg_dimension:.4f}")
        print(f"Theoretical dimension: 1.2618")
        print(f"Average error: {abs(avg_dimension - 1.2618):.4f}")
    
    return results