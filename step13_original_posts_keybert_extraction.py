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

def extract_replies_recursive(comment, replies_list, level=0, max_level=None):
    """
    Recursively extract replies from nested comment structure
    
    Args:
        comment: Comment object/dict
        replies_list: List to store extracted replies
        level: Current nesting level
        max_level: Maximum level to extract (None for unlimited)
    """
    if max_level is not None and level > max_level:
        return
    
    # Extract reply text
    reply_text = ""
    if isinstance(comment, dict):
        if 'body' in comment:
            reply_text = comment['body']
        elif 'text' in comment:
            reply_text = comment['text']
        elif 'content' in comment:
            reply_text = comment['content']
    
    if reply_text and reply_text.strip():
        replies_list.append({
            'reply': reply_text.strip(),
            'level': level
        })
    
    # Look for nested replies
    replies_key = None
    if isinstance(comment, dict):
        # Common keys for nested replies
        for key in ['replies', 'children', 'comments', 'responses']:
            if key in comment and comment[key]:
                replies_key = key
                break
    
    if replies_key and isinstance(comment[replies_key], list):
        for reply in comment[replies_key]:
            extract_replies_recursive(reply, replies_list, level + 1, max_level)

def main():
    # Read the CSV file
    df = pd.read_csv('original_posts_fractal_analysis.csv')

    # Display basic information about the dataset
    print("Dataset Info:")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFractal dimension statistics:")
    print(df['fractal_dimension'].describe())

    # Sort by fractal_dimension to get highest and lowest values
    df_sorted = df.sort_values('fractal_dimension', ascending=False)

    # Get top 42 posts with highest fractal dimension
    top_10_highest = df_sorted.head(42)[['post_id', 'fractal_dimension', 'conversation_type', 'thread_title']]

    # Get top 10 posts with lowest fractal dimension
    top_10_lowest = df_sorted.tail(42)[['post_id', 'fractal_dimension', 'conversation_type', 'thread_title']]

    print("\n" + "="*80)
    print("TOP 42 POSTS WITH HIGHEST FRACTAL DIMENSION")
    print("="*80)
    for i, row in top_10_highest.iterrows():
        print(f"{i + 1:2d}. Post ID: {row['post_id']}")
        print(f"    Fractal Dimension: {row['fractal_dimension']:.6f}")
        print(f"    Conversation Type: {row['conversation_type']}")
        print(f"    Thread Title: {row['thread_title']}")
        print()

    print("\n" + "="*80)
    print("TOP 42 POSTS WITH LOWEST FRACTAL DIMENSION")
    print("="*80)
    # Sort the lowest in ascending order for better readability
    top_10_lowest_sorted = top_10_lowest.sort_values('fractal_dimension', ascending=True)
    for i, row in top_10_lowest_sorted.iterrows():
        print(f"{i + 1:2d}. Post ID: {row['post_id']}")
        print(f"    Fractal Dimension: {row['fractal_dimension']:.6f}")
        print(f"    Conversation Type: {row['conversation_type']}")
        print(f"    Thread Title: {row['thread_title']}")
        print()

    # Extract post IDs
    highest_post_ids = top_10_highest['post_id'].tolist()
    lowest_post_ids = top_10_lowest_sorted['post_id'].tolist()
    
    print(f"\nHighest post IDs: {highest_post_ids}")
    print(f"Lowest post IDs: {lowest_post_ids}")

    # Load conversations from JSON folder
    print("\n" + "="*80)
    print("LOADING CONVERSATIONS FROM JSON FOLDER")
    print("="*80)
    
    json_folder = "json_data"
    conversations = load_conversations_from_folder(json_folder)
    
    # Create a dictionary for quick lookup using post_id from JSON files
    conversations_dict = {conv['post_id']: conv for conv in conversations}
    
    print(f"Available JSON post IDs: {list(conversations_dict.keys())}")
    print(f"Target highest post IDs: {highest_post_ids}")
    print(f"Target lowest post IDs: {lowest_post_ids}")
    
    # Combine target post IDs
    target_post_ids = highest_post_ids + lowest_post_ids
    
    # Find matching JSON files for our target post IDs
    print("\n" + "="*80)
    print("MATCHING POST IDs WITH JSON FILES")
    print("="*80)
    
    matched_posts = []
    unmatched_posts = []
    
    for post_id in target_post_ids:
        # Look for exact prefix match in JSON filenames
        found_match = False
        for json_post_id in conversations_dict.keys():
            if json_post_id == post_id:  # Exact match with the prefix
                matched_posts.append((post_id, json_post_id))
                print(f"Exact match: {post_id}")
                found_match = True
                break
        
        if not found_match:
            unmatched_posts.append(post_id)
            print(f"No exact match found for: {post_id}")
    
    print(f"\nMatching summary:")
    print(f"  Matched: {len(matched_posts)} posts")
    print(f"  Unmatched: {len(unmatched_posts)} posts")
    if unmatched_posts:
        print(f"  Unmatched posts: {unmatched_posts}")
        print("  Available JSON post IDs:", list(conversations_dict.keys()))
    
    # Extract replies for target posts
    print("\n" + "="*80)
    print("EXTRACTING REPLIES")
    print("="*80)
    
    final_data = []
    
    for csv_post_id, json_post_id in matched_posts:
        print(f"\nProcessing post: {csv_post_id} (JSON: {json_post_id})")
        
        # Get fractal dimension info from CSV using the original post_id
        post_info = df[df['post_id'] == csv_post_id].iloc[0]
        fractal_dimension = post_info['fractal_dimension']
        
        # Determine fractal dimension type
        if csv_post_id in highest_post_ids:
            fractal_dimension_type = "highest"
        else:
            fractal_dimension_type = "lowest"
        
        # Get conversation type from original data
        conversation_type = post_info['conversation_type']
        
        # Get the conversation data using the JSON post_id
        if json_post_id in conversations_dict:
            conv = conversations_dict[json_post_id]
            replies_list = []
            
            # Extract replies from the JSON data (capped at level 4)
            data = conv['data']
            
            # Handle different JSON structures
            if isinstance(data, list):
                for item in data:
                    extract_replies_recursive(item, replies_list, max_level=4)
            elif isinstance(data, dict):
                extract_replies_recursive(data, replies_list, max_level=4)
            
            print(f"  Found {len(replies_list)} replies (up to level 4)")
            
            # Add each reply as a separate row
            for reply_info in replies_list:
                final_data.append({
                    'post_id': csv_post_id,  # Use original CSV post_id
                    'conversation_type': conversation_type,
                    'fractal_dimension': fractal_dimension,
                    'fractal_dimension_type': fractal_dimension_type,
                    'reply': reply_info['reply'],
                    'reply_level': reply_info['level']
                })
        else:
            print(f"  Error: JSON post_id {json_post_id} not found in conversations")
    
    # Handle unmatched posts
    for post_id in unmatched_posts:
        print(f"\nProcessing unmatched post: {post_id}")
        
        # Get fractal dimension info
        post_info = df[df['post_id'] == post_id].iloc[0]
        fractal_dimension = post_info['fractal_dimension']
        
        # Determine fractal dimension type
        if post_id in highest_post_ids:
            fractal_dimension_type = "highest"
        else:
            fractal_dimension_type = "lowest"
        
        # Get conversation type from original data
        conversation_type = post_info['conversation_type']
        
        print(f"  Warning: No JSON file found for post_id: {post_id}")
        # Add a row indicating no replies found
        final_data.append({
            'post_id': post_id,
            'conversation_type': conversation_type,
            'fractal_dimension': fractal_dimension,
            'fractal_dimension_type': fractal_dimension_type,
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
    summary = final_df.groupby(['fractal_dimension_type', 'post_id']).size().reset_index(name='reply_count')
    print(f"\nReplies per post:")
    for _, row in summary.iterrows():
        print(f"  {row['fractal_dimension_type'].upper()} - {row['post_id']}: {row['reply_count']} replies")
    
    # Save to CSV
    output_filename = 'fractal_dimension_replies_analysis.csv'
    final_df.to_csv(output_filename, index=False)
    print(f"\nResults saved to: {output_filename}")
    
    return final_df, highest_post_ids, lowest_post_ids


# Run the analysis
if __name__ == "__main__":
    final_results, highest_ids, lowest_ids = main()