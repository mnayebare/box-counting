import json
import numpy as np
import re
import os
import csv
from datetime import datetime
from collections import defaultdict, Counter

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
            if "con" in post_id.lower():
                conversation_type = "controversial"
            elif "tec" in post_id.lower():
                conversation_type = "technical"
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

def cap_conversation_at_level4(data):
    """
    Cap conversation data at level 4 (depth 4) by removing deeper comments
    
    Parameters:
    - data: Original conversation data dictionary
    
    Returns:
    - Modified conversation data with depth capped at 4
    """
    def filter_comments_by_depth(comments, max_depth=4):
        """Recursively filter comments to only include those at or below max_depth"""
        filtered_comments = []
        
        for comment in comments:
            if comment.get('depth', 0) <= max_depth:
                # Create a copy of the comment
                filtered_comment = comment.copy()
                
                # If this comment has replies, filter them too
                if comment.get('replies') and comment.get('depth', 0) < max_depth:
                    filtered_comment['replies'] = filter_comments_by_depth(comment['replies'], max_depth)
                else:
                    # Remove replies if we're at max depth or no replies exist
                    filtered_comment['replies'] = []
                
                filtered_comments.append(filtered_comment)
        
        return filtered_comments
    
    # Create a copy of the original data
    capped_data = data.copy()
    
    # Filter comments to only include those at depth <= 4
    if 'comments' in capped_data:
        capped_data['comments'] = filter_comments_by_depth(capped_data['comments'], max_depth=4)
    
    # Update title to indicate capping
    original_title = capped_data.get('post_title', 'Unknown')
    capped_data['post_title'] = f"{original_title} (Level-4 Capped)"
    
    return capped_data

def find_deepest_subtree(comments_data, min_depth_threshold=2, max_depth_cap=4):
    """
    Find the single subtree with maximum depth that exceeds the minimum threshold
    and is capped at max_depth_cap
    
    Parameters:
    - comments_data: The comments structure from JSON
    - min_depth_threshold: Minimum depth to consider a subtree significant
    - max_depth_cap: Maximum depth to analyze (4 for level-4 cap)
    
    Returns:
    - Single subtree dictionary with metadata, or None if no qualifying subtree found
    """
    deepest_subtree = None
    max_depth_span = 0
    
    def calculate_subtree_depth(comment, depth_cap):
        """Calculate the maximum depth of a subtree starting from this comment, capped at depth_cap"""
        if not comment.get('replies') or comment['depth'] >= depth_cap:
            return min(comment['depth'], depth_cap)
        
        max_child_depth = comment['depth']
        for reply in comment['replies']:
            if reply['depth'] <= depth_cap:
                child_depth = calculate_subtree_depth(reply, depth_cap)
                max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth
    
    def extract_subtree_data(comment, subtree_id, depth_cap):
        """Extract a subtree as a standalone conversation structure, capped at depth_cap"""
        # Create a new data structure with this comment as the root
        subtree_data = {
            'post_title': f"Deepest Subtree from: {comment.get('author', 'Unknown')} (Level-4 Capped)",
            'post_timestamp': comment.get('timestamp', 'Unknown'),
            'subtree_id': subtree_id,
            'original_depth': comment['depth'],
            'comments': []
        }
        
        # Normalize depths and apply cap (make this comment depth 0)
        def normalize_comment_depth(comment_node, depth_offset, max_depth):
            if comment_node['depth'] > max_depth:
                return None
                
            normalized_comment = comment_node.copy()
            normalized_comment['depth'] = comment_node['depth'] - depth_offset
            normalized_comment['original_depth'] = comment_node['depth']  # Keep track of original depth
            
            if comment_node.get('replies') and comment_node['depth'] < max_depth:
                normalized_comment['replies'] = []
                for reply in comment_node['replies']:
                    if reply['depth'] <= max_depth:
                        normalized_reply = normalize_comment_depth(reply, depth_offset, max_depth)
                        if normalized_reply:
                            normalized_comment['replies'].append(normalized_reply)
            else:
                normalized_comment['replies'] = []
            
            return normalized_comment
        
        # Add the normalized subtree
        normalized_root = normalize_comment_depth(comment, comment['depth'], depth_cap)
        if normalized_root:
            subtree_data['comments'] = [normalized_root]
        
        return subtree_data
    
    def traverse_and_find_deepest(comments, parent_id="root"):
        """Traverse comments and find the single deepest subtree"""
        nonlocal deepest_subtree, max_depth_span
        
        for i, comment in enumerate(comments):
            # Skip if this comment is already beyond our cap
            if comment['depth'] > max_depth_cap:
                continue
                
            # Calculate the depth of this subtree (capped)
            subtree_max_depth = calculate_subtree_depth(comment, max_depth_cap)
            subtree_depth_span = subtree_max_depth - comment['depth']
            
            # If this subtree is deeper than our current deepest and meets threshold
            if subtree_depth_span >= min_depth_threshold and subtree_depth_span > max_depth_span:
                subtree_id = f"deepest_subtree_{comment.get('author', 'unknown')}_level4"
                
                deepest_subtree = {
                    'subtree_id': subtree_id,
                    'root_author': comment.get('author', 'Unknown'),
                    'root_depth': comment['depth'],
                    'max_depth': subtree_max_depth,
                    'depth_span': subtree_depth_span,
                    'root_score': comment.get('score', 0),
                    'root_timestamp': comment.get('timestamp', 'Unknown'),
                    'subtree_data': extract_subtree_data(comment, subtree_id, max_depth_cap)
                }
                
                max_depth_span = subtree_depth_span
                print(f"New deepest subtree found: {subtree_id} (depth span: {subtree_depth_span}, max depth: {subtree_max_depth})")
            
            # Recursively check replies
            if comment.get('replies'):
                traverse_and_find_deepest(comment['replies'], f"{parent_id}_{i}")
    
    # Start traversal
    traverse_and_find_deepest(comments_data.get('comments', []))
    
    return deepest_subtree

def clean_reddit_text(text):
    """
    Clean Reddit comment text by removing formatting and special characters
    
    Parameters:
    - text: Raw comment text
    
    Returns:
    - Cleaned text ready for word counting
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove Reddit-specific formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold formatting
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove italic formatting
    text = re.sub(r'~~(.*?)~~', r'\1', text)      # Remove strikethrough
    text = re.sub(r'\^(.*?)\^', r'\1', text)      # Remove superscript
    text = re.sub(r'`(.*?)`', r'\1', text)        # Remove inline code
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)  # Remove code blocks
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    
    # Remove Reddit usernames and subreddit mentions
    text = re.sub(r'/u/\S+', '', text)
    text = re.sub(r'u/\S+', '', text)
    text = re.sub(r'/r/\S+', '', text)
    text = re.sub(r'r/\S+', '', text)
    
    # Remove quote markers
    text = re.sub(r'^&gt;.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^>.*$', '', text, flags=re.MULTILINE)
    
    # Remove extra whitespace and newlines
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove non-alphabetic characters except spaces and basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:]', '', text)
    
    return text.strip()

def count_words_and_stats(text):
    """
    Count words and calculate text statistics
    
    Parameters:
    - text: Cleaned comment text
    
    Returns:
    - Dictionary with word count, character count, and average word length
    """
    if not text:
        return {
            'word_count': 0,
            'character_count': 0,
            'avg_word_length': 0.0,
            'cleaned_text': ""
        }
    
    # Split into words (alphanumeric sequences)
    words = re.findall(r'\b\w+\b', text.lower())
    word_count = len(words)
    character_count = len(text)
    
    # Calculate average word length
    if word_count > 0:
        avg_word_length = sum(len(word) for word in words) / word_count
    else:
        avg_word_length = 0.0
    
    return {
        'word_count': word_count,
        'character_count': character_count,
        'avg_word_length': round(avg_word_length, 2),
        'cleaned_text': text[:100] + "..." if len(text) > 100 else text  # Sample for verification
    }

def analyze_subtree_comment_word_lengths(comments, subtree_info, parent_post_id, conversation_type, results_list, comment_counter):
    """
    Recursively analyze word lengths in all subtree comments
    
    Parameters:
    - comments: List of comment dictionaries from subtree
    - subtree_info: Subtree metadata dictionary
    - parent_post_id: Original post ID
    - conversation_type: Parent conversation type
    - results_list: List to append results to
    - comment_counter: Counter for unique comment IDs
    """
    for comment in comments:
        # Generate unique comment ID for subtree
        comment_counter[0] += 1
        comment_id = f"{subtree_info['subtree_id']}_comment_{comment_counter[0]:04d}"
        
        # Extract comment text
        comment_text = comment.get('body', '') or comment.get('text', '')
        
        # Clean and analyze text
        cleaned_text = clean_reddit_text(comment_text)
        text_stats = count_words_and_stats(cleaned_text)
        
        # Create result record with subtree-specific fields
        result = {
            'parent_post_id': parent_post_id,
            'conversation_type': conversation_type,
            'subtree_id': subtree_info['subtree_id'],
            'subtree_root_author': subtree_info['root_author'],
            'subtree_root_depth': subtree_info['root_depth'],
            'subtree_max_depth': subtree_info['max_depth'],
            'subtree_depth_span': subtree_info['depth_span'],
            'comment_id': comment_id,
            'author': comment.get('author', 'Unknown'),
            'normalized_depth': comment.get('depth', 0),  # Depth within subtree (0-based)
            'original_depth': comment.get('original_depth', comment.get('depth', 0)),  # Original depth in full conversation
            'timestamp': comment.get('timestamp', 'Unknown'),
            'score': comment.get('score', 0),
            'word_count': text_stats['word_count'],
            'character_count': text_stats['character_count'],
            'avg_word_length': text_stats['avg_word_length'],
            'cleaned_text_sample': text_stats['cleaned_text'],
            'depth_cap_applied': 'Yes',
            'max_depth_analyzed': 4
        }
        
        results_list.append(result)
        
        # Recursively process replies
        if comment.get('replies'):
            analyze_subtree_comment_word_lengths(comment['replies'], subtree_info, parent_post_id, conversation_type, results_list, comment_counter)

def calculate_subtree_summary(subtree_results, subtree_info, parent_post_id, conversation_type):
    """
    Calculate summary statistics for a subtree
    
    Parameters:
    - subtree_results: List of comment analysis results for this subtree
    - subtree_info: Subtree metadata
    - parent_post_id: Original post ID
    - conversation_type: Parent conversation type
    
    Returns:
    - Dictionary with subtree-level statistics
    """
    if not subtree_results:
        return None
    
    # Calculate statistics
    word_counts = [c['word_count'] for c in subtree_results]
    char_counts = [c['character_count'] for c in subtree_results]
    avg_word_lengths = [c['avg_word_length'] for c in subtree_results if c['avg_word_length'] > 0]
    normalized_depths = [c['normalized_depth'] for c in subtree_results]
    original_depths = [c['original_depth'] for c in subtree_results]
    
    # Group by normalized depth for analysis
    normalized_depth_stats = defaultdict(list)
    for comment in subtree_results:
        normalized_depth_stats[comment['normalized_depth']].append(comment['word_count'])
    
    # Group by original depth for comparison
    original_depth_stats = defaultdict(list)
    for comment in subtree_results:
        original_depth_stats[comment['original_depth']].append(comment['word_count'])
    
    summary = {
        'parent_post_id': parent_post_id,
        'conversation_type': conversation_type,
        'subtree_id': subtree_info['subtree_id'],
        'subtree_root_author': subtree_info['root_author'],
        'subtree_root_depth': subtree_info['root_depth'],
        'subtree_max_depth': subtree_info['max_depth'],
        'subtree_depth_span': subtree_info['depth_span'],
        'subtree_root_score': subtree_info['root_score'],
        'subtree_root_timestamp': subtree_info['root_timestamp'],
        'total_comments': len(subtree_results),
        'total_words': sum(word_counts),
        'total_characters': sum(char_counts),
        'avg_words_per_comment': round(np.mean(word_counts), 2) if word_counts else 0,
        'median_words_per_comment': round(np.median(word_counts), 2) if word_counts else 0,
        'std_words_per_comment': round(np.std(word_counts), 2) if word_counts else 0,
        'min_words_per_comment': min(word_counts) if word_counts else 0,
        'max_words_per_comment': max(word_counts) if word_counts else 0,
        'avg_word_length_overall': round(np.mean(avg_word_lengths), 2) if avg_word_lengths else 0,
        'max_normalized_depth_reached': max(normalized_depths) if normalized_depths else 0,
        'max_original_depth_reached': max(original_depths) if original_depths else 0,
        'comments_by_normalized_depth': dict(Counter(normalized_depths)),
        'comments_by_original_depth': dict(Counter(original_depths)),
        'avg_words_by_normalized_depth': {depth: round(np.mean(words), 2) for depth, words in normalized_depth_stats.items()},
        'avg_words_by_original_depth': {depth: round(np.mean(words), 2) for depth, words in original_depth_stats.items()},
        'depth_cap_applied': 'Yes',
        'max_depth_analyzed': 4
    }
    
    return summary

def save_subtree_detailed_results_to_csv(results_data, filename='level4_subtree_word_length_detailed_results.csv'):
    """
    Save detailed subtree comment-level analysis results to CSV
    
    Parameters:
    - results_data: List of subtree comment analysis results
    - filename: Name of the CSV file to save
    """
    if not results_data:
        print("No subtree detailed data to save to CSV")
        return
    
    fieldnames = [
        'parent_post_id', 'conversation_type', 'subtree_id', 'subtree_root_author',
        'subtree_root_depth', 'subtree_max_depth', 'subtree_depth_span',
        'comment_id', 'author', 'normalized_depth', 'original_depth', 'timestamp', 'score',
        'word_count', 'character_count', 'avg_word_length', 'cleaned_text_sample',
        'depth_cap_applied', 'max_depth_analyzed'
    ]
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results_data:
            writer.writerow(result)
    
    print(f"Subtree detailed analysis results saved to: {filename}")
    print(f"Total subtree comment records written: {len(results_data)}")

def save_subtree_summary_results_to_csv(summary_data, filename='level4_subtree_word_length_summary_results.csv'):
    """
    Save subtree-level summary results to CSV
    
    Parameters:
    - summary_data: List of subtree summary results
    - filename: Name of the CSV file to save
    """
    if not summary_data:
        print("No subtree summary data to save to CSV")
        return
    
    fieldnames = [
        'parent_post_id', 'conversation_type', 'subtree_id', 'subtree_root_author',
        'subtree_root_depth', 'subtree_max_depth', 'subtree_depth_span',
        'subtree_root_score', 'subtree_root_timestamp',
        'total_comments', 'total_words', 'total_characters',
        'avg_words_per_comment', 'median_words_per_comment', 'std_words_per_comment',
        'min_words_per_comment', 'max_words_per_comment', 'avg_word_length_overall',
        'max_normalized_depth_reached', 'max_original_depth_reached',
        'norm_comments_at_depth_0', 'norm_comments_at_depth_1', 'norm_comments_at_depth_2',
        'norm_comments_at_depth_3', 'norm_comments_at_depth_4',
        'orig_comments_at_depth_0', 'orig_comments_at_depth_1', 'orig_comments_at_depth_2',
        'orig_comments_at_depth_3', 'orig_comments_at_depth_4',
        'norm_avg_words_depth_0', 'norm_avg_words_depth_1', 'norm_avg_words_depth_2',
        'norm_avg_words_depth_3', 'norm_avg_words_depth_4',
        'orig_avg_words_depth_0', 'orig_avg_words_depth_1', 'orig_avg_words_depth_2',
        'orig_avg_words_depth_3', 'orig_avg_words_depth_4',
        'depth_cap_applied', 'max_depth_analyzed'
    ]
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for summary in summary_data:
            # Flatten the nested dictionaries for CSV
            row = summary.copy()
            
            # Add normalized depth-specific comment counts
            norm_comments_by_depth = summary.get('comments_by_normalized_depth', {})
            for depth in range(5):  # 0-4
                row[f'norm_comments_at_depth_{depth}'] = norm_comments_by_depth.get(depth, 0)
            
            # Add original depth-specific comment counts
            orig_comments_by_depth = summary.get('comments_by_original_depth', {})
            for depth in range(5):  # 0-4
                row[f'orig_comments_at_depth_{depth}'] = orig_comments_by_depth.get(depth, 0)
            
            # Add normalized depth-specific average word counts
            norm_avg_words_by_depth = summary.get('avg_words_by_normalized_depth', {})
            for depth in range(5):  # 0-4
                row[f'norm_avg_words_depth_{depth}'] = norm_avg_words_by_depth.get(depth, 0)
            
            # Add original depth-specific average word counts
            orig_avg_words_by_depth = summary.get('avg_words_by_original_depth', {})
            for depth in range(5):  # 0-4
                row[f'orig_avg_words_depth_{depth}'] = orig_avg_words_by_depth.get(depth, 0)
            
            # Remove the nested dictionaries
            row.pop('comments_by_normalized_depth', None)
            row.pop('comments_by_original_depth', None)
            row.pop('avg_words_by_normalized_depth', None)
            row.pop('avg_words_by_original_depth', None)
            
            writer.writerow(row)
    
    print(f"Subtree summary analysis results saved to: {filename}")
    print(f"Total subtree records written: {len(summary_data)}")

def get_max_depth_from_data(data):
    """Calculate the maximum depth in the conversation data"""
    max_depth = 0
    
    def find_max_depth(comments):
        nonlocal max_depth
        for comment in comments:
            current_depth = comment.get('depth', 0)
            max_depth = max(max_depth, current_depth)
            if comment.get('replies'):
                find_max_depth(comment['replies'])
    
    find_max_depth(data.get('comments', []))
    return max_depth

def print_subtree_analysis(all_detailed_results):
    """Print detailed analysis of subtree word patterns"""
    print(f"\n{'='*60}")
    print("SUBTREE DEPTH-WISE ANALYSIS")
    print(f"{'='*60}")
    
    # Group by parent conversation type and normalized depth
    richly_by_norm_depth = defaultdict(list)
    poorly_by_norm_depth = defaultdict(list)
    richly_by_orig_depth = defaultdict(list)
    poorly_by_orig_depth = defaultdict(list)
    
    for result in all_detailed_results:
        norm_depth = result['normalized_depth']
        orig_depth = result['original_depth']
        word_count = result['word_count']
        
        if result['conversation_type'] == 'contoverisal':
            richly_by_norm_depth[norm_depth].append(word_count)
            richly_by_orig_depth[orig_depth].append(word_count)
        elif result['conversation_type'] == 'poorly technical':
            poorly_by_norm_depth[norm_depth].append(word_count)
            poorly_by_orig_depth[orig_depth].append(word_count)
    
    print("\nSUBTREE ANALYSIS by NORMALIZED DEPTH (within subtree):")
    print("\nCONTROVERSIAL subtrees:")
    for depth in sorted(richly_by_norm_depth.keys()):
        words = richly_by_norm_depth[depth]
        avg_words = round(np.mean(words), 2)
        count = len(words)
        print(f"  Normalized Depth {depth}: {count} comments, avg {avg_words} words/comment")
    
    print("\nTECHNICAL subtrees:")
    for depth in sorted(poorly_by_norm_depth.keys()):
        words = poorly_by_norm_depth[depth]
        avg_words = round(np.mean(words), 2)
        count = len(words)
        print(f"  Normalized Depth {depth}: {count} comments, avg {avg_words} words/comment")
    
    print("\nSUBTREE ANALYSIS by ORIGINAL DEPTH (in full conversation):")
    print("\nCONTROVERSIAL subtrees:")
    for depth in sorted(richly_by_orig_depth.keys()):
        words = richly_by_orig_depth[depth]
        avg_words = round(np.mean(words), 2)
        count = len(words)
        print(f"  Original Depth {depth}: {count} comments, avg {avg_words} words/comment")
    
    print("\nTECHNICAL subtrees:")
    for depth in sorted(poorly_by_orig_depth.keys()):
        words = poorly_by_orig_depth[depth]
        avg_words = round(np.mean(words), 2)
        count = len(words)
        print(f"  Original Depth {depth}: {count} comments, avg {avg_words} words/comment")

# MAIN EXECUTION
if __name__ == "__main__":
    print("Running Reddit Thread Subtree Word Length Analysis - Level-4 Capped...")
    
    # Specify the folder containing your JSON files
    json_data_folder = "json_data"  # Change this path if your folder is named differently
    
    # Load all conversations from the folder
    print(f"\n=== Loading conversations from '{json_data_folder}' folder ===")
    conversations = load_conversations_from_folder(json_data_folder)
    
    if not conversations:
        print("No conversation files found. Please check the folder path and file naming.")
        exit(1)
    
    # Store all results
    all_subtree_detailed_results = []
    all_subtree_summary_results = []
    
    # Track subtree statistics
    subtrees_found = 0
    subtrees_analyzed = 0
    
    # Process each conversation to find and analyze subtrees
    for i, conv in enumerate(conversations, 1):
        print(f"\n{'='*60}")
        print(f"Processing {i}/{len(conversations)}: {conv['post_id']} ({conv['conversation_type']}) - LEVEL-4 CAPPED")
        print(f"{'='*60}")
        
        try:
            # Apply level-4 cap to the conversation data
            original_data = conv['data']
            capped_data = cap_conversation_at_level4(original_data)
            
            print(f"Applied Level-4 cap to conversation data")
            original_max_depth = get_max_depth_from_data(original_data)
            capped_max_depth = get_max_depth_from_data(capped_data)
            print(f"Original max depth: {original_max_depth}, Capped max depth: {capped_max_depth}")
            
            # Find deepest subtree (now with level-4 cap)
            print(f"\n--- Finding Deepest Subtree (Level-4 Capped) ---")
            deepest_subtree = find_deepest_subtree(capped_data, max_depth_cap=4)
            
            if deepest_subtree:
                subtrees_found += 1
                print(f"Found qualifying subtree with depth span: {deepest_subtree['depth_span']}")
                print(f"Subtree root: {deepest_subtree['root_author']} at original depth {deepest_subtree['root_depth']}")
                print(f"Subtree spans from depth {deepest_subtree['root_depth']} to {deepest_subtree['max_depth']}")
                
                # Analyze word lengths in this subtree
                print(f"\n--- Analyzing Word Lengths in Subtree Comments ---")
                comment_counter = [0]  # Use list to modify in nested function
                subtree_results = []
                
                analyze_subtree_comment_word_lengths(
                    deepest_subtree['subtree_data'].get('comments', []),
                    deepest_subtree,
                    conv['post_id'],
                    conv['conversation_type'],
                    subtree_results,
                    comment_counter
                )
                
                if subtree_results:
                    subtrees_analyzed += 1
                    
                    # Add to overall results
                    all_subtree_detailed_results.extend(subtree_results)
                    
                    # Calculate subtree summary
                    summary = calculate_subtree_summary(subtree_results, deepest_subtree, conv['post_id'], conv['conversation_type'])
                    if summary:
                        all_subtree_summary_results.append(summary)
                    
                    # Print summary for this subtree
                    total_comments = len(subtree_results)
                    total_words = sum(r['word_count'] for r in subtree_results)
                    avg_words = round(total_words / total_comments, 2) if total_comments > 0 else 0
                    
                    print(f"\n--- Results for Subtree {deepest_subtree['subtree_id']} ---")
                    print(f"Total comments analyzed: {total_comments}")
                    print(f"Total words: {total_words}")
                    print(f"Average words per comment: {avg_words}")
                    print(f"Normalized depth range: 0 to {max([r['normalized_depth'] for r in subtree_results])}")
                    print(f"Original depth range: {min([r['original_depth'] for r in subtree_results])} to {max([r['original_depth'] for r in subtree_results])}")
                else:
                    print("No comments found in subtree for analysis")
            else:
                print("No qualifying subtree found (minimum depth threshold not met or insufficient depth after capping)")
                
        except Exception as e:
            print(f"Error processing {conv['post_id']}: {e}")
            continue
    
    # Generate final statistics
    print(f"\n{'='*60}")
    print("FINAL SUMMARY - LEVEL-4 CAPPED SUBTREE WORD LENGTH ANALYSIS")
    print(f"{'='*60}")
    
    total_conversations = len(conversations)
    total_subtree_comments = len(all_subtree_detailed_results)
    total_subtree_words = sum(r['word_count'] for r in all_subtree_detailed_results)
    
    # Separate by parent conversation type
    richly_branching_subtrees = [r for r in all_subtree_detailed_results if r['conversation_type'] == 'richly branching']
    poorly_branching_subtrees = [r for r in all_subtree_detailed_results if r['conversation_type'] == 'poorly branching']
    
    print(f"Total conversations processed: {total_conversations}")
    print(f"Subtrees found: {subtrees_found}")
    print(f"Subtrees analyzed: {subtrees_analyzed}")
    print(f"Total subtree comments analyzed: {total_subtree_comments}")
    print(f"Total words in subtrees: {total_subtree_words}")
    
    print(f"\nBy parent conversation type:")
    print(f"  From richly branching: {len(richly_branching_subtrees)} comments, {sum(r['word_count'] for r in richly_branching_subtrees)} words")
    print(f"  From poorly branching: {len(poorly_branching_subtrees)} comments, {sum(r['word_count'] for r in poorly_branching_subtrees)} words")
    
    if richly_branching_subtrees:
        rich_avg = round(np.mean([r['word_count'] for r in richly_branching_subtrees]), 2)
        print(f"  Richly branching subtrees avg words/comment: {rich_avg}")
    
    if poorly_branching_subtrees:
        poor_avg = round(np.mean([r['word_count'] for r in poorly_branching_subtrees]), 2)
        print(f"  Poorly branching subtrees avg words/comment: {poor_avg}")
    
    # Print detailed subtree analysis
    if all_subtree_detailed_results:
        print_subtree_analysis(all_subtree_detailed_results)
    
    # Analyze subtree characteristics
    print(f"\n{'='*60}")
    print("SUBTREE CHARACTERISTICS ANALYSIS")
    print(f"{'='*60}")
    
    if all_subtree_summary_results:
        # Analyze by parent conversation type
        richly_summaries = [s for s in all_subtree_summary_results if s['conversation_type'] == 'richly branching']
        poorly_summaries = [s for s in all_subtree_summary_results if s['conversation_type'] == 'poorly branching']
        
        print(f"\nSubtree depth spans:")
        if richly_summaries:
            rich_spans = [s['subtree_depth_span'] for s in richly_summaries]
            print(f"  Richly branching: avg span = {round(np.mean(rich_spans), 2)}, spans = {rich_spans}")
        
        if poorly_summaries:
            poor_spans = [s['subtree_depth_span'] for s in poorly_summaries]
            print(f"  Poorly branching: avg span = {round(np.mean(poor_spans), 2)}, spans = {poor_spans}")
        
        print(f"\nSubtree root depths (where subtrees start in original conversation):")
        if richly_summaries:
            rich_roots = [s['subtree_root_depth'] for s in richly_summaries]
            print(f"  Richly branching: avg root depth = {round(np.mean(rich_roots), 2)}, root depths = {rich_roots}")
        
        if poorly_summaries:
            poor_roots = [s['subtree_root_depth'] for s in poorly_summaries]
            print(f"  Poorly branching: avg root depth = {round(np.mean(poor_roots), 2)}, root depths = {poor_roots}")
    
    # Save results to CSV files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    detailed_csv_filename = f"level4_subtree_word_length_detailed_results.csv"
    save_subtree_detailed_results_to_csv(all_subtree_detailed_results, detailed_csv_filename)
    
    # Save summary results
    summary_csv_filename = f"level4_subtree_word_length_summary_results.csv"
    save_subtree_summary_results_to_csv(all_subtree_summary_results, summary_csv_filename)
    
    print(f"\nAll Level-4 capped subtree word length analysis complete!")
    print(f"Results saved to:")
    print(f"  - Detailed (per subtree comment): {detailed_csv_filename}")
    print(f"  - Summary (per subtree): {summary_csv_filename}")
    print(f"\nNote: All analysis was performed with depth capped at Level-4 (max depth = 4)")
    print(f"CSV files include both normalized depths (within subtree) and original depths (in full conversation)")
    
    # Print sample statistics
    print(f"\nSample statistics across all subtree comments:")
    if all_subtree_detailed_results:
        word_counts = [r['word_count'] for r in all_subtree_detailed_results]
        normalized_depths = [r['normalized_depth'] for r in all_subtree_detailed_results]
        original_depths = [r['original_depth'] for r in all_subtree_detailed_results]
        
        print(f"  Word count range: {min(word_counts)} - {max(word_counts)}")
        print(f"  Word count average: {round(np.mean(word_counts), 2)}")
        print(f"  Word count median: {round(np.median(word_counts), 2)}")
        print(f"  Word count std dev: {round(np.std(word_counts), 2)}")
        print(f"  Normalized depth range: {min(normalized_depths)} - {max(normalized_depths)}")
        print(f"  Original depth range: {min(original_depths)} - {max(original_depths)}")
        
        # Print examples of different depth patterns
        print(f"\nExample depth patterns:")
        unique_subtrees = list(set([r['subtree_id'] for r in all_subtree_detailed_results]))
        for subtree_id in unique_subtrees[:3]:  # Show first 3 subtrees
            subtree_comments = [r for r in all_subtree_detailed_results if r['subtree_id'] == subtree_id]
            if subtree_comments:
                print(f"  {subtree_id}:")
                print(f"    Normalized depths: {[r['normalized_depth'] for r in subtree_comments]}")
                print(f"    Original depths: {[r['original_depth'] for r in subtree_comments]}")
                print(f"    Word counts: {[r['word_count'] for r in subtree_comments]}")
        
        # Print top examples by word count
        sorted_by_words = sorted(all_subtree_detailed_results, key=lambda x: x['word_count'])
        
        print(f"\nShortest subtree comments:")
        for i in range(min(3, len(sorted_by_words))):
            comment = sorted_by_words[i]
            print(f"  {comment['word_count']} words (norm depth {comment['normalized_depth']}, orig depth {comment['original_depth']}): '{comment['cleaned_text_sample'][:50]}...'")
        
        print(f"\nLongest subtree comments:")
        for i in range(min(3, len(sorted_by_words))):
            comment = sorted_by_words[-(i+1)]
            print(f"  {comment['word_count']} words (norm depth {comment['normalized_depth']}, orig depth {comment['original_depth']}): '{comment['cleaned_text_sample'][:50]}...'")
    
    # Compare subtree patterns to help with analysis
    print(f"\n{'='*60}")
    print("SUBTREE vs PARENT CONVERSATION COMPARISON INSIGHTS")
    print(f"{'='*60}")
    
    if all_subtree_summary_results:
        print(f"\nKey insights for analysis:")
        print(f"1. Subtrees were found in {subtrees_found}/{total_conversations} conversations")
        print(f"2. Richly branching conversations had {len(richly_summaries)} qualifying subtrees")
        print(f"3. Poorly branching conversations had {len(poorly_summaries)} qualifying subtrees")
        
        if richly_summaries and poorly_summaries:
            rich_avg_words = np.mean([s['avg_words_per_comment'] for s in richly_summaries])
            poor_avg_words = np.mean([s['avg_words_per_comment'] for s in poorly_summaries])
            print(f"4. Average words per comment in subtrees:")
            print(f"   - Richly branching subtrees: {round(rich_avg_words, 2)} words/comment")
            print(f"   - Poorly branching subtrees: {round(poor_avg_words, 2)} words/comment")
            
            if rich_avg_words > poor_avg_words:
                print(f"   → Richly branching subtrees have {round(rich_avg_words - poor_avg_words, 2)} more words per comment on average")
            else:
                print(f"   → Poorly branching subtrees have {round(poor_avg_words - rich_avg_words, 2)} more words per comment on average")
    
    print(f"\n" + "="*60)
    print("SUBTREE ANALYSIS COMPLETE!")
    print(f"="*60)
    print(f"Use the CSV files to compare:")
    print(f"- Word patterns at different subtree depths (normalized vs original)")
    print(f"- Linguistic differences between richly vs poorly branching subtrees")
    print(f"- How deep conversations evolve linguistically")
    print(f"- Subtree characteristics vs their parent conversations")