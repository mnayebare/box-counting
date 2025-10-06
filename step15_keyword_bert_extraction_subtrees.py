import pandas as pd
import json
import os

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
                conversation_type = "tehnical"
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

def extract_replies_from_subtree(comment, replies_list, target_author, target_timestamp, 
                               level=0, max_level=None, found_subtree=False):
    """
    Recursively extract replies from a specific subtree
    
    Args:
        comment: Comment object/dict
        replies_list: List to store extracted replies
        target_author: Author of the subtree root
        target_timestamp: Timestamp of the subtree root (for matching)
        level: Current nesting level
        max_level: Maximum level to extract from subtree
        found_subtree: Whether we've found the target subtree
    
    Returns:
        bool: Whether the subtree was found and processed
    """
    if max_level is not None and level > max_level:
        return found_subtree
    
    # Extract comment info
    reply_text = ""
    comment_author = ""
    comment_timestamp = ""
    
    if isinstance(comment, dict):
        # Extract text
        if 'body' in comment:
            reply_text = comment['body']
        elif 'text' in comment:
            reply_text = comment['text']
        elif 'content' in comment:
            reply_text = comment['content']
            
        # Extract author
        if 'author' in comment:
            comment_author = comment['author']
        elif 'username' in comment:
            comment_author = comment['username']
        elif 'user' in comment:
            comment_author = comment['user']
            
        # Extract timestamp (various possible formats)
        for ts_key in ['timestamp', 'created_utc', 'created', 'time', 'date']:
            if ts_key in comment and comment[ts_key]:
                comment_timestamp = str(comment[ts_key])
                break
    
    # Check if this is our target subtree root
    is_subtree_root = False
    if (not found_subtree and 
        comment_author == target_author and
        target_timestamp in comment_timestamp):  # Flexible timestamp matching
        found_subtree = True
        is_subtree_root = True
        print(f"    Found subtree root by '{target_author}' at level {level}")
    
    # Add reply if we're in the target subtree
    if found_subtree and reply_text and reply_text.strip():
        replies_list.append({
            'reply': reply_text.strip(),
            'level': level,
            'author': comment_author,
            'is_subtree_root': is_subtree_root,
            'timestamp': comment_timestamp
        })
    
    # Look for nested replies
    replies_key = None
    if isinstance(comment, dict):
        for key in ['replies', 'children', 'comments', 'responses']:
            if key in comment and comment[key]:
                replies_key = key
                break
    
    # Continue searching in replies
    if replies_key and isinstance(comment[replies_key], list):
        for reply in comment[replies_key]:
            found_subtree = extract_replies_from_subtree(
                reply, replies_list, target_author, target_timestamp, 
                level + 1, max_level, found_subtree
            )
            # If we found the subtree and we're not in it yet, stop searching other branches
            if found_subtree and not is_subtree_root and level == 0:
                break
    
    return found_subtree

def extract_all_replies_recursive(comment, replies_list, level=0, max_level=None):
    """
    Fallback function to extract all replies when subtree matching fails
    """
    if max_level is not None and level > max_level:
        return
    
    # Extract reply text and author
    reply_text = ""
    comment_author = ""
    
    if isinstance(comment, dict):
        # Extract text
        if 'body' in comment:
            reply_text = comment['body']
        elif 'text' in comment:
            reply_text = comment['text']
        elif 'content' in comment:
            reply_text = comment['content']
            
        # Extract author
        if 'author' in comment:
            comment_author = comment['author']
        elif 'username' in comment:
            comment_author = comment['username']
        elif 'user' in comment:
            comment_author = comment['user']
    
    if reply_text and reply_text.strip():
        replies_list.append({
            'reply': reply_text.strip(),
            'level': level,
            'author': comment_author,
            'is_subtree_root': False,
            'timestamp': ""
        })
    
    # Look for nested replies
    replies_key = None
    if isinstance(comment, dict):
        for key in ['replies', 'children', 'comments', 'responses']:
            if key in comment and comment[key]:
                replies_key = key
                break
    
    if replies_key and isinstance(comment[replies_key], list):
        for reply in comment[replies_key]:
            extract_all_replies_recursive(reply, replies_list, level + 1, max_level)

def main():
    # Read the subtree fractal analysis CSV file
    df = pd.read_csv('subtree_fractal_analysis.csv')

    # Display basic information about the dataset
    print("Dataset Info:")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFractal dimension statistics:")
    print(df['fractal_dimension'].describe())

    # Sort by fractal_dimension to get highest and lowest values
    df_sorted = df.sort_values('fractal_dimension', ascending=False)

    # Get top 40 posts with highest fractal dimension
    top_10_highest = df_sorted.head(40)[['post_id', 'fractal_dimension', 'conversation_type', 'thread_title']]

    # Get top 40 posts with lowest fractal dimension
    top_10_lowest = df_sorted.tail(40)[['post_id', 'fractal_dimension', 'conversation_type', 'thread_title']]

    print("\n" + "="*80)
    print("TOP 10 POSTS WITH HIGHEST FRACTAL DIMENSION")
    print("="*80)
    for i, (idx, row) in enumerate(top_10_highest.iterrows()):
        print(f"{i + 1:2d}. Post ID: {row['post_id']}")
        print(f"    Fractal Dimension: {row['fractal_dimension']:.6f}")
        print(f"    Conversation Type: {row['conversation_type']}")
        print(f"    Thread Title: {row['thread_title']}")
        print()

    print("\n" + "="*80)
    print("TOP 10 POSTS WITH LOWEST FRACTAL DIMENSION")
    print("="*80)
    # Sort the lowest in ascending order for better readability
    top_10_lowest_sorted = top_10_lowest.sort_values('fractal_dimension', ascending=True)
    for i, (idx, row) in enumerate(top_10_lowest_sorted.iterrows()):
        print(f"{i + 1:2d}. Post ID: {row['post_id']}")
        print(f"    Fractal Dimension: {row['fractal_dimension']:.6f}")
        print(f"    Conversation Type: {row['conversation_type']}")
        print(f"    Thread Title: {row['thread_title']}")
        print()

    # Extract post IDs for the highest and lowest fractal dimension posts
    highest_post_ids = top_10_highest['post_id'].unique().tolist()
    lowest_post_ids = top_10_lowest_sorted['post_id'].unique().tolist()
    
    print(f"\nHighest fractal dimension post IDs: {highest_post_ids}")
    print(f"Lowest fractal dimension post IDs: {lowest_post_ids}")

    # Load conversations from JSON folder
    print("\n" + "="*80)
    print("LOADING CONVERSATIONS FROM JSON FOLDER")
    print("="*80)
    
    json_folder = "json_data"
    conversations = load_conversations_from_folder(json_folder)
    
    # Create a dictionary for quick lookup using post_id from JSON files
    conversations_dict = {conv['post_id']: conv for conv in conversations}
    
    print(f"Available JSON post IDs: {list(conversations_dict.keys())}")
    print(f"Target highest fractal dimension post IDs: {highest_post_ids}")
    print(f"Target lowest fractal dimension post IDs: {lowest_post_ids}")
    
    # Combine target post IDs
    target_post_ids = highest_post_ids + lowest_post_ids
    
    # Find matching JSON files for our target post IDs
    print("\n" + "="*80)
    print("MATCHING POST IDs WITH JSON FILES")
    print("="*80)
    
    matched_posts = []
    unmatched_posts = []
    
    for post_id in target_post_ids:
        # Look for exact match in JSON filenames
        if post_id in conversations_dict:
            matched_posts.append(post_id)
            print(f"Exact match: {post_id}")
        else:
            unmatched_posts.append(post_id)
            print(f"No match found for: {post_id}")
    
    print(f"\nMatching summary:")
    print(f"  Matched: {len(matched_posts)} posts")
    print(f"  Unmatched: {len(unmatched_posts)} posts")
    
    # Extract replies for matched posts
    print("\n" + "="*80)
    print("EXTRACTING REPLIES FROM HIGHEST DEPTH SUBTREES")
    print("="*80)
    
    final_data = []
    
    for post_id in matched_posts:
        # Determine if this is a highest or lowest fractal dimension post
        fractal_dimension_category = ""
        if post_id in highest_post_ids:
            fractal_dimension_category = "highest"
        elif post_id in lowest_post_ids:
            fractal_dimension_category = "lowest"
        
        print(f"\nProcessing {fractal_dimension_category} fractal dimension post: {post_id}")
        
        # Get all subtrees for this post_id from CSV
        post_subtrees = df[df['post_id'] == post_id]
        
        if len(post_subtrees) == 0:
            print(f"  Warning: No subtrees found in CSV for post_id: {post_id}")
            continue
        
        # Find the subtree with the highest max_depth for this post
        highest_depth_subtree = post_subtrees.loc[post_subtrees['subtree_max_depth'].idxmax()]
        
        subtree_id = highest_depth_subtree['subtree_id']
        max_depth = highest_depth_subtree['subtree_max_depth']
        fractal_dimension = highest_depth_subtree['fractal_dimension']
        conversation_type = highest_depth_subtree['conversation_type']
        subtree_root_author = highest_depth_subtree['subtree_root_author']
        subtree_root_timestamp = str(highest_depth_subtree['subtree_root_timestamp'])
        thread_title = highest_depth_subtree['thread_title']
        
        print(f"  Selected highest depth subtree: {subtree_id}")
        print(f"  Max depth: {max_depth}")
        print(f"  Root author: {subtree_root_author}")
        print(f"  Root timestamp: {subtree_root_timestamp}")
        print(f"  Fractal dimension: {fractal_dimension:.6f}")
        
        # Get the conversation data and extract replies from the specific subtree
        conv = conversations_dict[post_id]
        replies_list = []
        
        # Limit extraction to level 4 maximum
        extraction_depth = min(max_depth, 4)
        print(f"  Limiting extraction to level {extraction_depth} (subtree max: {max_depth})")
        
        # Try to extract replies from the specific subtree
        data = conv['data']
        subtree_found = False
        
        # Handle different JSON structures
        if isinstance(data, list):
            for item in data:
                found = extract_replies_from_subtree(
                    item, replies_list, subtree_root_author, subtree_root_timestamp, 
                    max_level=extraction_depth, found_subtree=False
                )
                if found:
                    subtree_found = True
                    break
        elif isinstance(data, dict):
            subtree_found = extract_replies_from_subtree(
                data, replies_list, subtree_root_author, subtree_root_timestamp,
                max_level=extraction_depth, found_subtree=False
            )
        
        # If we couldn't find the specific subtree, fall back to extracting all replies
        if not subtree_found or len(replies_list) == 0:
            print(f"  Warning: Could not locate specific subtree, extracting all replies up to depth {extraction_depth}")
            replies_list = []
            if isinstance(data, list):
                for item in data:
                    extract_all_replies_recursive(item, replies_list, max_level=extraction_depth)
            elif isinstance(data, dict):
                extract_all_replies_recursive(data, replies_list, max_level=extraction_depth)
        
        print(f"  Extracted {len(replies_list)} replies from subtree (up to level {extraction_depth})")
        
        # Add each reply as a separate row
        for reply_info in replies_list:
            final_data.append({
                'post_id': post_id,
                'conversation_type': conversation_type,
                'fractal_dimension': fractal_dimension,
                'fractal_dimension_type': fractal_dimension_category,
                'reply': reply_info['reply'],
                'reply_level': reply_info['level']
            })
    
    # Handle unmatched posts
    print("\n" + "="*40)
    print("HANDLING UNMATCHED POSTS")
    print("="*40)
    
    for post_id in unmatched_posts:
        # Determine fractal dimension category
        fractal_dimension_category = ""
        if post_id in highest_post_ids:
            fractal_dimension_category = "highest"
        elif post_id in lowest_post_ids:
            fractal_dimension_category = "lowest"
        
        print(f"Processing unmatched {fractal_dimension_category} post: {post_id}")
        
        # Get the highest depth subtree for this post from CSV
        post_subtrees = df[df['post_id'] == post_id]
        
        if len(post_subtrees) > 0:
            highest_depth_subtree = post_subtrees.loc[post_subtrees['subtree_max_depth'].idxmax()]
            
            final_data.append({
                'post_id': post_id,
                'conversation_type': highest_depth_subtree['conversation_type'],
                'fractal_dimension': highest_depth_subtree['fractal_dimension'],
                'fractal_dimension_type': fractal_dimension_category,
                'reply': "NO_JSON_FILE_FOUND",
                'reply_level': 0
            })
    
    # Create final DataFrame
    final_df = pd.DataFrame(final_data)
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Total replies extracted: {len(final_df)}")
    print(f"Columns: {list(final_df.columns)}")
    
    # Display summary
    if len(final_df) > 0:
        summary = final_df.groupby(['fractal_dimension_type', 'post_id']).size().reset_index(name='reply_count')
        
        print(f"\nReplies per post by fractal dimension type:")
        for _, row in summary.iterrows():
            post_info = final_df[final_df['post_id'] == row['post_id']].iloc[0]
            print(f"  {row['fractal_dimension_type'].upper()} - {row['post_id']}: "
                  f"{row['reply_count']} replies (fractal_dim: {post_info['fractal_dimension']:.4f})")
        
        # Show category breakdown
        print(f"\nFractal dimension type breakdown:")
        category_breakdown = final_df['fractal_dimension_type'].value_counts()
        print(category_breakdown)
        
        # Show reply level distribution
        print(f"\nReply level distribution:")
        level_distribution = final_df['reply_level'].value_counts().sort_index()
        print(level_distribution)
    
    # Save to CSV with exact column order
    final_df = final_df[['post_id', 'conversation_type', 'fractal_dimension', 'fractal_dimension_type', 'reply', 'reply_level']]
    output_filename = 'subtree_fractal_dimension_replies_analysis.csv'
    final_df.to_csv(output_filename, index=False)
    print(f"\n✓ Results saved to: {output_filename}")
    
    return final_df, highest_post_ids, lowest_post_ids


# Run the analysis
if __name__ == "__main__":
    final_results, highest_ids, lowest_ids = main()