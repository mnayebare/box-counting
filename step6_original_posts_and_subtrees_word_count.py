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

def analyze_comment_word_lengths(comments, post_id, conversation_type, results_list, comment_counter):
    """
    Recursively analyze word lengths in all comments
    
    Parameters:
    - comments: List of comment dictionaries
    - post_id: Post identifier
    - conversation_type: Type of conversation (richly/poorly branching)
    - results_list: List to append results to
    - comment_counter: Counter for unique comment IDs
    """
    for comment in comments:
        # Generate unique comment ID
        comment_counter[0] += 1
        comment_id = f"{post_id}_comment_{comment_counter[0]:04d}"
        
        # Extract comment text
        comment_text = comment.get('body', '') or comment.get('text', '')
        
        # Clean and analyze text
        cleaned_text = clean_reddit_text(comment_text)
        text_stats = count_words_and_stats(cleaned_text)
        
        # Create result record
        result = {
            'post_id': post_id,
            'conversation_type': conversation_type,
            'comment_id': comment_id,
            'author': comment.get('author', 'Unknown'),
            'depth': comment.get('depth', 0),
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
            analyze_comment_word_lengths(comment['replies'], post_id, conversation_type, results_list, comment_counter)

def calculate_conversation_summary(comment_results, post_id, conversation_type):
    """
    Calculate summary statistics for a conversation
    
    Parameters:
    - comment_results: List of comment analysis results for this conversation
    - post_id: Post identifier
    - conversation_type: Type of conversation
    
    Returns:
    - Dictionary with conversation-level statistics
    """
    if not comment_results:
        return None
    
    # Filter results for this conversation
    conv_comments = [r for r in comment_results if r['post_id'] == post_id]
    
    if not conv_comments:
        return None
    
    # Calculate statistics
    word_counts = [c['word_count'] for c in conv_comments]
    char_counts = [c['character_count'] for c in conv_comments]
    avg_word_lengths = [c['avg_word_length'] for c in conv_comments if c['avg_word_length'] > 0]
    depths = [c['depth'] for c in conv_comments]
    
    # Group by depth for analysis
    depth_stats = defaultdict(list)
    for comment in conv_comments:
        depth_stats[comment['depth']].append(comment['word_count'])
    
    summary = {
        'post_id': post_id,
        'conversation_type': conversation_type,
        'total_comments': len(conv_comments),
        'total_words': sum(word_counts),
        'total_characters': sum(char_counts),
        'avg_words_per_comment': round(np.mean(word_counts), 2) if word_counts else 0,
        'median_words_per_comment': round(np.median(word_counts), 2) if word_counts else 0,
        'std_words_per_comment': round(np.std(word_counts), 2) if word_counts else 0,
        'min_words_per_comment': min(word_counts) if word_counts else 0,
        'max_words_per_comment': max(word_counts) if word_counts else 0,
        'avg_word_length_overall': round(np.mean(avg_word_lengths), 2) if avg_word_lengths else 0,
        'max_depth_reached': max(depths) if depths else 0,
        'comments_by_depth': dict(Counter(depths)),
        'avg_words_by_depth': {depth: round(np.mean(words), 2) for depth, words in depth_stats.items()},
        'depth_cap_applied': 'Yes',
        'max_depth_analyzed': 4
    }
    
    return summary

def save_detailed_results_to_csv(results_data, filename='level4_original_posts_length_detailed_results.csv'):
    """
    Save detailed comment-level analysis results to CSV
    
    Parameters:
    - results_data: List of comment analysis results
    - filename: Name of the CSV file to save
    """
    if not results_data:
        print("No detailed data to save to CSV")
        return
    
    fieldnames = [
        'post_id', 'conversation_type', 'comment_id', 'author', 'depth', 'timestamp', 'score',
        'word_count', 'character_count', 'avg_word_length', 'cleaned_text_sample',
        'depth_cap_applied', 'max_depth_analyzed'
    ]
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results_data:
            writer.writerow(result)
    
    print(f"Detailed analysis results saved to: {filename}")
    print(f"Total comment records written: {len(results_data)}")

def save_summary_results_to_csv(summary_data, filename='level4_original_posts_length_summary_results.csv'):
    """
    Save conversation-level summary results to CSV
    
    Parameters:
    - summary_data: List of conversation summary results
    - filename: Name of the CSV file to save
    """
    if not summary_data:
        print("No summary data to save to CSV")
        return
    
    fieldnames = [
        'post_id', 'conversation_type', 'total_comments', 'total_words', 'total_characters',
        'avg_words_per_comment', 'median_words_per_comment', 'std_words_per_comment',
        'min_words_per_comment', 'max_words_per_comment', 'avg_word_length_overall',
        'max_depth_reached', 'comments_at_depth_0', 'comments_at_depth_1', 'comments_at_depth_2',
        'comments_at_depth_3', 'comments_at_depth_4', 'avg_words_depth_0', 'avg_words_depth_1',
        'avg_words_depth_2', 'avg_words_depth_3', 'avg_words_depth_4',
        'depth_cap_applied', 'max_depth_analyzed'
    ]
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for summary in summary_data:
            # Flatten the nested dictionaries for CSV
            row = summary.copy()
            
            # Add depth-specific comment counts
            comments_by_depth = summary.get('comments_by_depth', {})
            for depth in range(5):  # 0-4
                row[f'comments_at_depth_{depth}'] = comments_by_depth.get(depth, 0)
            
            # Add depth-specific average word counts
            avg_words_by_depth = summary.get('avg_words_by_depth', {})
            for depth in range(5):  # 0-4
                row[f'avg_words_depth_{depth}'] = avg_words_by_depth.get(depth, 0)
            
            # Remove the nested dictionaries
            row.pop('comments_by_depth', None)
            row.pop('avg_words_by_depth', None)
            
            writer.writerow(row)
    
    print(f"Summary analysis results saved to: {filename}")
    print(f"Total conversation records written: {len(summary_data)}")

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

def print_depth_analysis(all_detailed_results):
    """Print detailed analysis by depth and conversation type"""
    print(f"\n{'='*60}")
    print("DEPTH-WISE ANALYSIS")
    print(f"{'='*60}")
    
    # Group by conversation type and depth
    richly_by_depth = defaultdict(list)
    poorly_by_depth = defaultdict(list)
    
    for result in all_detailed_results:
        depth = result['depth']
        word_count = result['word_count']
        
        if result['conversation_type'] == 'richly branching':
            richly_by_depth[depth].append(word_count)
        elif result['conversation_type'] == 'poorly branching':
            poorly_by_depth[depth].append(word_count)
    
    print("\nRICHLY BRANCHING conversations:")
    for depth in sorted(richly_by_depth.keys()):
        words = richly_by_depth[depth]
        avg_words = round(np.mean(words), 2)
        count = len(words)
        print(f"  Depth {depth}: {count} comments, avg {avg_words} words/comment")
    
    print("\nPOORLY BRANCHING conversations:")
    for depth in sorted(poorly_by_depth.keys()):
        words = poorly_by_depth[depth]
        avg_words = round(np.mean(words), 2)
        count = len(words)
        print(f"  Depth {depth}: {count} comments, avg {avg_words} words/comment")
    
    # Compare averages across types
    print(f"\nCOMPARISON by depth:")
    all_depths = set(list(richly_by_depth.keys()) + list(poorly_by_depth.keys()))
    for depth in sorted(all_depths):
        rich_avg = round(np.mean(richly_by_depth[depth]), 2) if richly_by_depth[depth] else 0
        poor_avg = round(np.mean(poorly_by_depth[depth]), 2) if poorly_by_depth[depth] else 0
        rich_count = len(richly_by_depth[depth])
        poor_count = len(poorly_by_depth[depth])
        
        print(f"  Depth {depth}: Richly={rich_avg} words ({rich_count} comments) vs Poorly={poor_avg} words ({poor_count} comments)")

# MAIN EXECUTION
if __name__ == "__main__":
    print("Running Reddit Thread Word Length Analysis - Level-4 Capped...")
    
    # Specify the folder containing your JSON files
    json_data_folder = "json_data"  # Change this path if your folder is named differently
    
    # Load all conversations from the folder
    print(f"\n=== Loading conversations from '{json_data_folder}' folder ===")
    conversations = load_conversations_from_folder(json_data_folder)
    
    if not conversations:
        print("No conversation files found. Please check the folder path and file naming.")
        exit(1)
    
    # Store all results
    all_detailed_results = []
    all_summary_results = []
    
    # Process each conversation
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
            
            # Analyze word lengths in this conversation
            print(f"\n--- Analyzing Word Lengths in Comments ---")
            comment_counter = [0]  # Use list to modify in nested function
            conversation_results = []
            
            analyze_comment_word_lengths(
                capped_data.get('comments', []), 
                conv['post_id'], 
                conv['conversation_type'], 
                conversation_results, 
                comment_counter
            )
            
            # Add to overall results
            all_detailed_results.extend(conversation_results)
            
            # Calculate conversation summary
            summary = calculate_conversation_summary(conversation_results, conv['post_id'], conv['conversation_type'])
            if summary:
                all_summary_results.append(summary)
            
            # Print summary for this conversation
            total_comments = len(conversation_results)
            total_words = sum(r['word_count'] for r in conversation_results)
            avg_words = round(total_words / total_comments, 2) if total_comments > 0 else 0
            
            print(f"\n--- Results for {conv['post_id']} ---")
            print(f"Total comments analyzed: {total_comments}")
            print(f"Total words: {total_words}")
            print(f"Average words per comment: {avg_words}")
            print(f"Depth range: 0 to {capped_max_depth}")
            
        except Exception as e:
            print(f"Error processing {conv['post_id']}: {e}")
            continue
    
    # Generate final statistics
    print(f"\n{'='*60}")
    print("FINAL SUMMARY - LEVEL-4 CAPPED WORD LENGTH ANALYSIS")
    print(f"{'='*60}")
    
    total_conversations = len(conversations)
    total_comments = len(all_detailed_results)
    total_words = sum(r['word_count'] for r in all_detailed_results)
    
    # Separate by conversation type
    richly_branching = [r for r in all_detailed_results if r['conversation_type'] == 'richly branching']
    poorly_branching = [r for r in all_detailed_results if r['conversation_type'] == 'poorly branching']
    
    print(f"Total conversations processed: {total_conversations}")
    print(f"Total comments analyzed: {total_comments}")
    print(f"Total words counted: {total_words}")
    print(f"\nBy conversation type:")
    print(f"  Richly branching: {len(richly_branching)} comments, {sum(r['word_count'] for r in richly_branching)} words")
    print(f"  Poorly branching: {len(poorly_branching)} comments, {sum(r['word_count'] for r in poorly_branching)} words")
    
    if richly_branching:
        rich_avg = round(np.mean([r['word_count'] for r in richly_branching]), 2)
        print(f"  Richly branching avg words/comment: {rich_avg}")
    
    if poorly_branching:
        poor_avg = round(np.mean([r['word_count'] for r in poorly_branching]), 2)
        print(f"  Poorly branching avg words/comment: {poor_avg}")
    
    # Print detailed depth analysis
    print_depth_analysis(all_detailed_results)
    
    # Save results to CSV files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    detailed_csv_filename = f"level4_original_posts_word_length_detailed_results.csv"
    save_detailed_results_to_csv(all_detailed_results, detailed_csv_filename)
    
    # Save summary results
    summary_csv_filename = f"level4_original_posts_word_length_summary_results.csv"
    save_summary_results_to_csv(all_summary_results, summary_csv_filename)
    
    print(f"\nAll Level-4 capped word length analysis complete!")
    print(f"Results saved to:")
    print(f"  - Detailed (per comment): {detailed_csv_filename}")
    print(f"  - Summary (per conversation): {summary_csv_filename}")
    print(f"\nNote: All analysis was performed with depth capped at Level-4 (max depth = 4)")
    print(f"CSV files include 'depth_cap_applied' and 'max_depth_analyzed' columns for tracking")
    
    # Print sample statistics
    print(f"\nSample statistics across all comments:")
    if all_detailed_results:
        word_counts = [r['word_count'] for r in all_detailed_results]
        print(f"  Word count range: {min(word_counts)} - {max(word_counts)}")
        print(f"  Word count average: {round(np.mean(word_counts), 2)}")
        print(f"  Word count median: {round(np.median(word_counts), 2)}")
        print(f"  Word count std dev: {round(np.std(word_counts), 2)}")
        
        # Print top 5 longest and shortest comments
        sorted_by_words = sorted(all_detailed_results, key=lambda x: x['word_count'])
        
        print(f"\nShortest comments:")
        for i in range(min(3, len(sorted_by_words))):
            comment = sorted_by_words[i]
            print(f"  {comment['word_count']} words: '{comment['cleaned_text_sample'][:50]}...'")
        
        print(f"\nLongest comments:")
        for i in range(min(3, len(sorted_by_words))):
            comment = sorted_by_words[-(i+1)]
            print(f"  {comment['word_count']} words: '{comment['cleaned_text_sample'][:50]}...'")
    
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print(f"="*60)