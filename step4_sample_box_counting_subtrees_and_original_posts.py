import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import csv

# Set matplotlib to use research paper-appropriate styling
plt.rcParams.update({
    'font.size': 14,           # Base font size
    'font.weight': 'bold',     # Bold fonts
    'axes.labelsize': 16,      # Axis label size
    'axes.titlesize': 18,      # Title size
    'axes.labelweight': 'bold', # Bold axis labels
    'axes.titleweight': 'bold', # Bold titles
    'xtick.labelsize': 12,     # X-tick label size
    'ytick.labelsize': 12,     # Y-tick label size
    'legend.fontsize': 14,     # Legend font size
    'axes.linewidth': 2,       # Axis line thickness
    'lines.linewidth': 2.5,    # Line thickness
    'lines.markersize': 6,     # Marker size
    'xtick.major.width': 2,    # X-tick thickness
    'ytick.major.width': 2,    # Y-tick thickness
    'axes.edgecolor': 'black', # Axis edge color
    'text.color': 'black',     # Text color
    'axes.labelcolor': 'black' # Label color
})

def load_conversations_from_folder(folder_path):
    """Load all conversation JSON files from the specified folder"""
    conversations = []
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return conversations
    
    # Updated filter to match the exact naming convention
    json_files = [f for f in os.listdir(folder_path) 
                  if f.endswith('_reddit_comments_with_time.json')]
    
    print(f"Found {len(json_files)} conversation files")
    
    for json_file in json_files:
        file_path = os.path.join(folder_path, json_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Extract post ID correctly - remove the suffix once
            post_id = json_file.replace('_reddit_comments_with_time.json', '')
            
            # Determine conversation type based on post_id
            conversation_type = ""
            if "hb" in post_id.lower():
                conversation_type = "richly branching"
            elif "lb" in post_id.lower():
                conversation_type = "poorly branching"
            else:
                conversation_type = "unknown"
            
            conversations.append({
                'post_id': post_id,
                'conversation_type': conversation_type,
                'data': data,
                'file_path': file_path
            })
            
            print(f"Loaded: {post_id} ({conversation_type})")
            
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    return conversations

def save_data_to_csv(results_data, filename='reddit_fractal_analysis_results.csv'):
    """
    Save all analysis results to a CSV file
    
    Parameters:
    - results_data: List of dictionaries containing analysis results
    - filename: Name of the CSV file to save
    """
    if not results_data:
        print("No data to save to CSV")
        return
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        # Get all unique fieldnames from all results
        fieldnames = set()
        for result in results_data:
            fieldnames.update(result.keys())
        
        # Define the order of columns
        ordered_fieldnames = [
            'post_id', 'conversation_type', 'fractal_dimension', 'total_contour_points',
            'sampled_points_count', 'x_min', 'x_max', 'y_min', 'y_max',
            'box_sizes_epsilon', 'box_counts', 'log_box_counts', 'log_inv_epsilon',
            'x_coordinates', 'y_coordinates', 'thread_title', 'time_span', 'subtree_id'
        ]
        
        # Add any remaining fieldnames that weren't explicitly ordered
        for field in fieldnames:
            if field not in ordered_fieldnames:
                ordered_fieldnames.append(field)
        
        writer = csv.DictWriter(csvfile, fieldnames=ordered_fieldnames)
        writer.writeheader()
        
        for result in results_data:
            writer.writerow(result)
    
    print(f"Analysis results saved to: {filename}")
    print(f"Total rows written: {len(results_data)}")

def process_conversation_data(conv, contour, fractal_dim, box_sizes, counts):
    """
    Process a single conversation's data and return structured results for CSV
    
    Returns a single dictionary with arrays stored as comma-separated strings
    """
    # Calculate log values
    log_counts = np.log(counts)
    log_inv_epsilon = np.log(1/box_sizes)
    
    # Get contour bounds
    x_min, x_max = contour[:, 0].min(), contour[:, 0].max()
    y_min, y_max = contour[:, 1].min(), contour[:, 1].max()
    
    # Sample contour points to reasonable size (~500 points max)
    sample_step = max(1, len(contour) // 500)
    sampled_indices = range(0, len(contour), sample_step)
    sampled_x = contour[sampled_indices, 0]
    sampled_y = contour[sampled_indices, 1]
    
    # Convert arrays to comma-separated strings
    result = {
        'post_id': conv['post_id'],
        'conversation_type': conv['conversation_type'],
        'fractal_dimension': fractal_dim,
        'total_contour_points': len(contour),
        'sampled_points_count': len(sampled_x),
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        
        # Box-counting arrays (stored as comma-separated strings)
        'box_sizes_epsilon': ','.join(map(str, box_sizes)),
        'box_counts': ','.join(map(str, counts)),
        'log_box_counts': ','.join(map(str, log_counts)),
        'log_inv_epsilon': ','.join(map(str, log_inv_epsilon)),
        
        # Coordinate arrays (stored as comma-separated strings)
        'x_coordinates': ','.join(map(str, sampled_x)),
        'y_coordinates': ','.join(map(str, sampled_y)),
        
        # Additional metadata
        'thread_title': conv['data'].get('post_title', 'Unknown'),
        'time_span': conv['data'].get('time_difference', 'Unknown')
    }
    
    return [result]  # Return as list for consistency with the main loop

def find_deepest_subtree(comments_data, min_depth_threshold=2):
    """
    Find the single subtree with maximum depth that exceeds the minimum threshold
    
    Parameters:
    - comments_data: The comments structure from JSON
    - min_depth_threshold: Minimum depth to consider a subtree significant (changed to 2)
    
    Returns:
    - Single subtree dictionary with metadata, or None if no qualifying subtree found
    """
    deepest_subtree = None
    max_depth_span = 0
    
    def calculate_subtree_depth(comment):
        """Calculate the maximum depth of a subtree starting from this comment"""
        if not comment.get('replies'):
            return comment['depth']
        
        max_child_depth = comment['depth']
        for reply in comment['replies']:
            child_depth = calculate_subtree_depth(reply)
            max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth
    
    def extract_subtree_data(comment, subtree_id):
        """Extract a subtree as a standalone conversation structure"""
        # Create a new data structure with this comment as the root
        subtree_data = {
            'post_title': f"Deepest Subtree from: {comment.get('author', 'Unknown')}",
            'post_timestamp': comment.get('timestamp', 'Unknown'),
            'subtree_id': subtree_id,
            'original_depth': comment['depth'],
            'comments': []
        }
        
        # Normalize depths (make this comment depth 0)
        def normalize_comment_depth(comment_node, depth_offset):
            normalized_comment = comment_node.copy()
            normalized_comment['depth'] = comment_node['depth'] - depth_offset
            
            if comment_node.get('replies'):
                normalized_comment['replies'] = []
                for reply in comment_node['replies']:
                    normalized_reply = normalize_comment_depth(reply, depth_offset)
                    normalized_comment['replies'].append(normalized_reply)
            
            return normalized_comment
        
        # Add the normalized subtree
        normalized_root = normalize_comment_depth(comment, comment['depth'])
        subtree_data['comments'] = [normalized_root]
        
        return subtree_data
    
    def traverse_and_find_deepest(comments, parent_id="root"):
        """Traverse comments and find the single deepest subtree"""
        nonlocal deepest_subtree, max_depth_span
        
        for i, comment in enumerate(comments):
            # Calculate the depth of this subtree
            subtree_max_depth = calculate_subtree_depth(comment)
            subtree_depth_span = subtree_max_depth - comment['depth']
            
            # If this subtree is deeper than our current deepest and meets threshold
            if subtree_depth_span >= min_depth_threshold and subtree_depth_span > max_depth_span:
                subtree_id = f"deepest_subtree_{comment.get('author', 'unknown')}"
                
                deepest_subtree = {
                    'subtree_id': subtree_id,
                    'root_author': comment.get('author', 'Unknown'),
                    'root_depth': comment['depth'],
                    'max_depth': subtree_max_depth,
                    'depth_span': subtree_depth_span,
                    'root_score': comment.get('score', 0),
                    'root_timestamp': comment.get('timestamp', 'Unknown'),
                    'subtree_data': extract_subtree_data(comment, subtree_id)
                }
                
                max_depth_span = subtree_depth_span
                print(f"New deepest subtree found: {subtree_id} (depth span: {subtree_depth_span}, max depth: {subtree_max_depth})")
            
            # Recursively check replies
            if comment.get('replies'):
                traverse_and_find_deepest(comment['replies'], f"{parent_id}_{i}")
    
    # Start traversal
    traverse_and_find_deepest(comments_data.get('comments', []))
    
    return deepest_subtree

def analyze_subtree(subtree_info, subtrees_contours_dir, subtrees_fractal_dir, parent_conversation_type, parent_post_id):
    """
    Analyze a single subtree and generate visualizations
    
    Returns processed data for CSV
    """
    subtree_id = subtree_info['subtree_id']
    subtree_data = subtree_info['subtree_data']
    
    print(f"\n--- Analyzing Subtree: {subtree_id} ---")
    print(f"Root author: {subtree_info['root_author']}")
    print(f"Depth span: {subtree_info['depth_span']} (from {subtree_info['root_depth']} to {subtree_info['max_depth']})")
    
    # Create extractor for this subtree
    extractor = ThreadContourExtractor(subtree_data)
    
    # Generate contour visualization using parent post ID for filename
    title_suffix = f" - Subtree ({subtree_info['root_author']}, span:{subtree_info['depth_span']})"
    contour_save_path = os.path.join(subtrees_contours_dir, f"subtree_contour_{parent_post_id}.png")
    
    contour, filename = extractor.visualize_contour(title_suffix, save_path=contour_save_path)
    
    # Prepare for fractal analysis
    raw_contour = extractor.prepare_for_fractal_analysis()
    
    # Calculate fractal dimension using parent post ID for filename
    fractal_save_path = os.path.join(subtrees_fractal_dir, f"subtree_fractal_{parent_post_id}.png")
    fractal_dim, box_sizes, counts, plot_file = box_counting_fractal_dimension(
        raw_contour, save_path=fractal_save_path)
    
    # Create a conversation-like structure for processing
    # Use parent_post_id instead of subtree_id to maintain connection
    conv_like = {
        'post_id': parent_post_id,  # Use parent post ID to match original posts
        'conversation_type': parent_conversation_type,  # Use parent's conversation type
        'data': subtree_data
    }
    
    # Process data for CSV
    processed_data = process_conversation_data(conv_like, raw_contour, fractal_dim, box_sizes, counts)[0]
    
    # Add subtree-specific metadata
    processed_data.update({
        'subtree_id': subtree_id,  # Keep the unique subtree identifier as additional field
        'subtree_root_author': subtree_info['root_author'],
        'subtree_root_depth': subtree_info['root_depth'],
        'subtree_max_depth': subtree_info['max_depth'],
        'subtree_depth_span': subtree_info['depth_span'],
        'subtree_root_score': subtree_info['root_score'],
        'subtree_root_timestamp': subtree_info['root_timestamp']
    })
    
    print(f"Subtree fractal dimension: {fractal_dim:.3f}")
    print(f"Subtree contour points: {len(raw_contour)}")
    
    return processed_data

def calculate_word_count_penalty(word_count):
    """
    Calculate penalty factor based on word count using new penalty range
    
    Parameters:
    - word_count: Number of words in the comment
    
    Returns:
    - penalty_factor: Multiplier between 0.4 and 1.0
      - 1.0 = no penalty (50+ words)
      - 0.7-1.0 = gentle penalty (20-49 words)  
      - 0.4-0.7 = moderate penalty (<20 words)
    """
    if word_count >= 50:
        return 1.0  # No penalty
    elif word_count >= 20:
        # Linear interpolation between 0.7 and 1.0 for 20-49 words
        return 0.7 + (word_count - 20) * (0.3 / 30)
    else:
        # Linear interpolation between 0.4 and 0.7 for 0-19 words
        return 0.4 + (word_count / 20) * 0.3

def count_words_in_comment(comment_text):
    """
    Count words in a comment, handling None/empty cases
    
    Parameters:
    - comment_text: The comment body text
    
    Returns:
    - word_count: Number of words (0 if empty/None)
    """
    if not comment_text or comment_text.strip() == "":
        return 0
    
    # Simple word counting - split by whitespace
    words = comment_text.strip().split()
    return len(words)

class ThreadContourExtractor:
    def __init__(self, json_data):
        """
        Initialize with Reddit thread JSON data
        """
        if isinstance(json_data, str):
            with open(json_data, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = json_data
        
        self.contour_points = []
    
    def extract_depth_oscillation_contour(self):
        """
        Create VERTICAL contour with word count penalty applied to indentation
        X-axis: Penalized conversation depth (reduced for low word count)
        Y-axis: Comment position (downward flow)
        """
        contour_points = []
        position_counter = [0]  # Use list to modify in nested function
        
        def traverse_vertically(comments, base_depth):
            for comment in comments:
                # Count words in this comment
                comment_text = comment.get('body', '') or comment.get('text', '') or comment.get('content', '')
                word_count = count_words_in_comment(comment_text)
                
                # Calculate penalty factor
                penalty_factor = calculate_word_count_penalty(word_count)
                
                # Apply penalty to depth (reduce indentation for low word count)
                penalized_depth = base_depth * penalty_factor
                
                # Add point: [penalized_depth, position] - SWAPPED AXES FOR VERTICAL FLOW
                contour_points.append([penalized_depth, position_counter[0]])
                position_counter[0] += 1
                
                # Process replies at increased depth (moving right)
                if comment.get('replies'):
                    # Add transition point going deeper (moving right)
                    transition_depth = penalized_depth + 0.5 * penalty_factor
                    contour_points.append([transition_depth, position_counter[0]])
                    position_counter[0] += 0.5
                    
                    # Process replies at deeper level (increment base_depth, not penalized_depth)
                    traverse_vertically(comment['replies'], base_depth + 1)
                    
                    # Add transition point coming back (moving left)
                    contour_points.append([transition_depth, position_counter[0]])
                    position_counter[0] += 0.5
        
        traverse_vertically(self.data['comments'], 0)
        self.contour_points = np.array(contour_points)
        return self.contour_points
    
    def visualize_contour(self, title_suffix="", save_path=None, figsize=(12, 8), dpi=300):
        """
        Visualize the extracted contour and save to file first - BLACK AND WHITE VERSION
        
        Parameters:
        - title_suffix: Additional text for plot title
        - save_path: Custom file path (optional)
        - figsize: Figure size in inches as (width, height)
        - dpi: Resolution in dots per inch
        """
        contour = self.extract_depth_oscillation_contour()
        
        # Create figure with white background
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        ax.set_facecolor('white')
        
        # Plot contour line in black with thick line
        ax.plot(contour[:, 0], contour[:, 1], 'k-', linewidth=3, alpha=0.8, label='Conversation Contour')
        
        # Plot points in dark gray, smaller size for research paper clarity
        ax.scatter(contour[:, 0], contour[:, 1], c='black', s=8, alpha=0.6, zorder=5)
        
        # Research paper appropriate labels - bold and clear
        ax.set_xlabel('Conversation Depth (Word Count Penalized)', fontweight='bold', fontsize=16)
        ax.set_ylabel('Comment Position (Chronological Order)', fontweight='bold', fontsize=16)
        ax.set_title(f'Conversation Thread Contour Analysis{title_suffix}', fontweight='bold', fontsize=18, pad=20)
        
        # Black grid for better readability
        ax.grid(True, alpha=0.3, color='black', linewidth=1)
        
        # Make axis spines thicker and black
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_color('black')
        
        # Bold tick labels
        ax.tick_params(axis='both', which='major', labelsize=12, width=2, length=6, colors='black')
        
        plt.tight_layout()
        
        # Save the figure FIRST
        if save_path is None:
            save_path = 'reddit_thread_contour_depth_oscillation.png'
        
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='black', linewidth=2)
        print(f"Contour visualization saved as: {save_path}")
        print(f"Image size: {figsize[0]}″ × {figsize[1]}″ at {dpi} DPI")
        print(f"Pixel dimensions: {int(figsize[0]*dpi)} × {int(figsize[1]*dpi)} pixels")
        
        # Close the figure to free memory when processing multiple files
        plt.close()
        
        return contour, save_path
    
    def prepare_for_fractal_analysis(self):
        """
        Prepare the raw contour data for box-counting fractal dimension analysis
        Returns the original contour without normalization
        """
        contour = self.extract_depth_oscillation_contour()
        
        print(f"Raw contour extracted using 'penalized depth_oscillation' method:")
        print(f"- Total points: {len(contour)}")
        print(f"- X range: [{contour[:, 0].min():.3f}, {contour[:, 0].max():.3f}]")
        print(f"- Y range: [{contour[:, 1].min():.3f}, {contour[:, 1].max():.3f}]")
        print(f"- Ready for box-counting analysis (using penalized coordinates)")
        
        return contour

def box_counting_fractal_dimension(contour_points, box_sizes=None, save_path=None, 
                                  figsize=(10, 6), dpi=300):
    """
    Calculate fractal dimension using box-counting method with SQUARE boxes - BLACK AND WHITE VERSION
    
    Parameters:
    - contour_points: Array of [x, y] coordinates
    - box_sizes: Array of box sizes (optional, auto-generated if None)
    - save_path: Custom file path (optional)
    - figsize: Figure size in inches as (width, height)
    - dpi: Resolution in dots per inch
    """
    # Step 1: Normalize contour to [0,1] × [0,1] unit square
    x_min, x_max = contour_points[:, 0].min(), contour_points[:, 0].max()
    y_min, y_max = contour_points[:, 1].min(), contour_points[:, 1].max()
    
    # Avoid division by zero
    x_range = x_max - x_min if x_max != x_min else 1.0
    y_range = y_max - y_min if y_max != y_min else 1.0
    
    # Normalize to [0,1] range
    x_norm = (contour_points[:, 0] - x_min) / x_range
    y_norm = (contour_points[:, 1] - y_min) / y_range
    normalized_points = np.column_stack([x_norm, y_norm])
    
    # Step 2: Generate box sizes (fractions of unit square)
    if box_sizes is None:
        # Better box sizes - not too small
        box_sizes = np.logspace(np.log10(0.05), np.log10(0.5), 15)
    
    counts = []
    
    for epsilon in box_sizes:
        # Create SQUARE grid with spacing epsilon
        occupied_boxes = set()
        
        for point in normalized_points:
            # Which square box (ε×ε) does this point fall into?
            box_x = int(point[0] / epsilon)
            box_y = int(point[1] / epsilon)
            occupied_boxes.add((box_x, box_y))
        
        counts.append(len(occupied_boxes))
    
    # Fit line to log-log plot
    log_sizes = np.log(1/box_sizes)
    log_counts = np.log(counts)
    
    # Linear regression
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    fractal_dimension = coeffs[0]
    
    # Create figure with white background for research paper
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')
    
    # Plot data points in black circles with thick lines
    ax.loglog(1/box_sizes, counts, 'ko-', linewidth=3, markersize=8, 
              markerfacecolor='black', markeredgecolor='black', markeredgewidth=2,
              label='Observed Data')
    
    # Plot fit line in thick black dashed line
    ax.loglog(1/box_sizes, np.exp(coeffs[1]) * (1/box_sizes)**coeffs[0], 
              'k--', linewidth=4, alpha=0.8,
              label=f'Linear Fit: D = {fractal_dimension:.3f}')
    
    # Research paper appropriate labels - bold and clear
    ax.set_xlabel('Inverse Box Size (1/ε)', fontweight='bold', fontsize=16)
    ax.set_ylabel('Number of Occupied Boxes N(ε)', fontweight='bold', fontsize=16)
    ax.set_title('Box-Counting Fractal Dimension Analysis', fontweight='bold', fontsize=18, pad=20)
    
    # Black legend with thick border
    legend = ax.legend(loc='upper left', fontsize=14, frameon=True, 
                      facecolor='white', edgecolor='black', linewidth=2)
    legend.get_frame().set_facecolor('white')
    
    # Black grid for better readability
    ax.grid(True, alpha=0.3, color='black', linewidth=1)
    
    # Make axis spines thicker and black
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')
    
    # Bold tick labels
    ax.tick_params(axis='both', which='major', labelsize=12, width=2, length=6, colors='black')
    ax.tick_params(axis='both', which='minor', width=1, length=4, colors='black')
    
    plt.tight_layout()
    
    # Save the fractal analysis plot FIRST
    if save_path is None:
        save_path = 'reddit_fractal_analysis.png'
    
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
               facecolor='white', edgecolor='black', linewidth=2)
    print(f"Fractal analysis plot saved as: {save_path}")
    print(f"Image size: {figsize[0]}″ × {figsize[1]}″ at {dpi} DPI")
    print(f"Pixel dimensions: {int(figsize[0]*dpi)} × {int(figsize[1]*dpi)} pixels")
    print(f"Box sizes used: {len(box_sizes)} square boxes from ε={box_sizes[0]:.3f} to ε={box_sizes[-1]:.3f}")
    print(f"Fractal dimension: {fractal_dimension:.3f}")
    
    # Close the figure to free memory when processing multiple files
    plt.close()
    
    return fractal_dimension, box_sizes, counts, save_path

# CORRECTED MAIN EXECUTION - Process all files from json_data folder
if __name__ == "__main__":
    print("Running Reddit Thread Fractal Analysis - Research Paper Black & White Version...")
    print("Features: Bold labels, thick lines, black and white color scheme, high contrast")
    
    # Specify the folder containing your JSON files
    json_data_folder = "json_data"  # Change this path if your folder is named differently
    
    # Load all conversations from the folder
    print(f"\n=== Loading conversations from '{json_data_folder}' folder ===")
    conversations = load_conversations_from_folder(json_data_folder)
    
    if not conversations:
        print("No conversation files found. Please check the folder path and file naming.")
        exit(1)
    
    # Create output directories for organized results
    contours_dir = "research_paper_contour_visualizations"
    fractal_dir = "research_paper_fractal_analysis"
    subtrees_contours_dir = "research_paper_subtree_contours"
    subtrees_fractal_dir = "research_paper_subtree_fractal_analysis"
    
    for directory in [contours_dir, fractal_dir, subtrees_contours_dir, subtrees_fractal_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Store all results for CSV export
    all_results = []
    
    # Process each conversation
    for i, conv in enumerate(conversations, 1):
        print(f"\n{'='*60}")
        print(f"Processing {i}/{len(conversations)}: {conv['post_id']} ({conv['conversation_type']})")
        print(f"{'='*60}")
        
        try:
            # Create extractor for this conversation
            extractor = ThreadContourExtractor(conv['data'])
            
            # Generate contour visualization
            title_suffix = f" - {conv['conversation_type'].title()} ({conv['post_id']})"
            contour_save_path = os.path.join(contours_dir, f"contour_{conv['post_id']}.png")
            
            print(f"\n--- Creating Research Paper Contour Visualization ---")
            contour, filename = extractor.visualize_contour(title_suffix, save_path=contour_save_path)
            
            # Prepare for fractal analysis
            print(f"\n--- Preparing for Fractal Analysis ---")
            raw_contour = extractor.prepare_for_fractal_analysis()
            
            # Calculate fractal dimension
            print(f"\n--- Calculating Fractal Dimension ---")
            fractal_save_path = os.path.join(fractal_dir, f"fractal_{conv['post_id']}.png")
            fractal_dim, box_sizes, counts, plot_file = box_counting_fractal_dimension(
                raw_contour, save_path=fractal_save_path)
            
            # Process data for CSV
            processed_data = process_conversation_data(conv, raw_contour, fractal_dim, box_sizes, counts)
            all_results.extend(processed_data)
            
            print(f"\n--- Results for {conv['post_id']} ---")
            print(f"Fractal dimension: {fractal_dim:.3f}")
            print(f"Contour points: {len(raw_contour)}")
            print(f"Thread title: {conv['data'].get('post_title', 'Unknown')}")
            
            # Find and analyze deepest subtree (now with minimum depth threshold of 2)
            print(f"\n--- Finding Deepest Subtree ---")
            deepest_subtree = find_deepest_subtree(conv['data'])
            
            if deepest_subtree:
                print(f"Found qualifying subtree with depth span: {deepest_subtree['depth_span']}")
                subtree_result = analyze_subtree(deepest_subtree, subtrees_contours_dir, subtrees_fractal_dir, conv['conversation_type'], conv['post_id'])
                all_results.append(subtree_result)
            else:
                print("No qualifying subtree found (minimum depth threshold not met)")
                
        except Exception as e:
            print(f"Error processing {conv['post_id']}: {e}")
            continue
    
    # Separate results into original posts and subtrees based on presence of subtree metadata
    original_posts = [result for result in all_results if 'subtree_root_author' not in result]
    subtree_results = [result for result in all_results if 'subtree_root_author' in result]
    
    print(f"\n{'='*60}")
    print("FINAL SUMMARY - RESEARCH PAPER VERSION")
    print(f"{'='*60}")
    print(f"Total conversations processed: {len(conversations)}")
    print(f"Total original post results: {len(original_posts)}")
    print(f"Total subtree results: {len(subtree_results)}")
    print(f"Total analysis results: {len(all_results)}")
    print(f"Images generated in BLACK & WHITE with BOLD LABELS for research paper use")
    
    # Save results to separate CSV files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save original posts results
    original_csv_filename = f"research_paper_original_posts_fractal_analysis.csv"
    if original_posts:
        save_data_to_csv(original_posts, original_csv_filename)
    else:
        print("No original post results to save")
    
    # Save subtree results
    subtree_csv_filename = f"research_paper_subtree_fractal_analysis.csv"
    if subtree_results:
        save_data_to_csv(subtree_results, subtree_csv_filename)
    else:
        print("No subtree results to save")
    
    print(f"\nResearch Paper Analysis Complete!")
    print(f"FORMATTING FEATURES:")
    print(f"  - All images: Black & white high contrast")
    print(f"  - Font sizes: 14-18pt with bold styling")
    print(f"  - Line thickness: 2.5-4pt for visibility")
    print(f"  - Grid lines: Black with transparency")
    print(f"  - Axis spines: 2pt thickness")
    print(f"  - DPI: 300 for publication quality")
    print(f"Word count penalty applied: <20 words = 0.4-0.7x depth, 20-49 words = 0.7-1.0x depth, 50+ words = 1.0x depth")
    print(f"\nResults saved to:")
    print(f"  - Original posts: {original_csv_filename}")
    print(f"  - Subtrees: {subtree_csv_filename}")
    print(f"\nResearch Paper Visualizations saved in:")
    print(f"  - Contours: {contours_dir}/")
    print(f"  - Fractal Analysis: {fractal_dir}/")
    print(f"  - Subtree Contours: {subtrees_contours_dir}/")
    print(f"  - Subtree Fractal Analysis: {subtrees_fractal_dir}/")