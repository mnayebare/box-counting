import json
import numpy as np
from datetime import datetime
import os
import csv
from collections import defaultdict
from pathlib import Path

def load_conversations_from_folder(folder_path):
    """Load all conversation JSON files from the specified folder"""
    conversations = []
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"Error: Folder '{folder_path}' not found.")
        return conversations
    
    # Updated filter to match the exact naming convention
    json_files = [f for f in folder_path.iterdir() 
                  if f.name.endswith('_reddit_comments_with_time.json')]
    
    print(f"Found {len(json_files)} conversation files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Extract post ID correctly - remove the suffix once
            post_id = json_file.name.replace('_reddit_comments_with_time.json', '')
            
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
                'file_path': str(json_file)
            })
            
            print(f"Loaded: {post_id} ({conversation_type})")
            
        except Exception as e:
            print(f"Error loading {json_file.name}: {e}")
            continue
    
    return conversations

def calculate_depth(comment, parent_depths):
    """Calculate the depth of a comment based on its parent"""
    if 'parent_id' not in comment or comment['parent_id'] is None:
        return 0
    
    parent_id = comment['parent_id']
    if parent_id in parent_depths:
        return parent_depths[parent_id] + 1
    else:
        return 1  # Fallback if parent not found

def analyze_conversation_structure(data):
    """Analyze the structure of a conversation thread"""
    # Handle the actual JSON structure - extract comments array
    if isinstance(data, dict) and 'comments' in data:
        comments = data['comments']
    elif isinstance(data, list):
        comments = data
    else:
        return {
            'total_comments': 0,
            'max_depth': 0,
            'depth_distribution': {},
            'deepest_comment': None,
            'subtree_stats': None,
            'tie_info': None
        }
    
    if not comments or not isinstance(comments, list):
        return {
            'total_comments': 0,
            'max_depth': 0,
            'depth_distribution': {},
            'deepest_comment': None,
            'subtree_stats': None,
            'tie_info': None
        }
    
    # Build parent-child relationships and calculate depths
    comment_depths = {}
    comments_by_id = {}
    children_count = defaultdict(int)
    
    # Flatten the nested comment structure since your data has nested replies
    def flatten_comments(comment_list, parent_id=None, current_depth=0):
        flat_comments = []
        for comment in comment_list:
            # Add depth information to the comment
            comment['calculated_depth'] = current_depth
            comment['parent_id'] = parent_id
            flat_comments.append(comment)
            
            # Process nested replies
            if 'replies' in comment and comment['replies']:
                flat_comments.extend(flatten_comments(
                    comment['replies'], 
                    comment.get('author'), 
                    current_depth + 1
                ))
        return flat_comments
    
    # Flatten all comments
    flat_comments = flatten_comments(comments)
    
    # Index all comments
    for comment in flat_comments:
        comment_id = comment.get('author')  # Using author as ID since that's what's available
        comments_by_id[comment_id] = comment
        comment_depths[comment_id] = comment.get('calculated_depth', 0)
        
        # Count children for branching analysis
        parent_id = comment.get('parent_id')
        if parent_id:
            children_count[parent_id] += 1
    
    # Find all comments with maximum depth
    max_depth = max(comment_depths.values()) if comment_depths else 0
    deepest_comments = [comment for comment in flat_comments 
                       if comment.get('calculated_depth') == max_depth]
    
    # Handle tie-breaking: largest_subtree first, then first_found
    selected_comment = None
    subtree_stats = None
    tie_info = {
        'max_depth_comment_count': len(deepest_comments),
        'tie_breaking_used': None
    }
    
    if deepest_comments:
        if len(deepest_comments) == 1:
            # No tie - simple case
            selected_comment = deepest_comments[0]
            tie_info['tie_breaking_used'] = 'none'
        else:
            # Multiple comments at max depth - use largest_subtree method
            subtree_sizes = []
            for comment in deepest_comments:
                temp_subtree = analyze_subtree(comment, flat_comments, comments_by_id)
                subtree_sizes.append((comment, temp_subtree['total_comments'], temp_subtree))
            
            # Sort by subtree size (descending)
            subtree_sizes.sort(key=lambda x: x[1], reverse=True)
            
            # Check if there are multiple subtrees with the same largest size
            largest_size = subtree_sizes[0][1]
            largest_subtrees = [item for item in subtree_sizes if item[1] == largest_size]
            
            if len(largest_subtrees) == 1:
                # Clear winner by subtree size
                selected_comment = largest_subtrees[0][0]
                subtree_stats = largest_subtrees[0][2]
                tie_info['tie_breaking_used'] = 'largest_subtree'
            else:
                # Tie in subtree sizes - use first_found
                selected_comment = largest_subtrees[0][0]
                subtree_stats = largest_subtrees[0][2]
                tie_info['tie_breaking_used'] = 'largest_subtree_then_first_found'
                tie_info['subtree_size_ties'] = len(largest_subtrees)
    
    # Calculate subtree stats if not already done
    if selected_comment and subtree_stats is None:
        subtree_stats = analyze_subtree(selected_comment, flat_comments, comments_by_id)
    
    # Calculate depth distribution
    depth_distribution = defaultdict(int)
    for depth in comment_depths.values():
        depth_distribution[depth] += 1
    
    return {
        'total_comments': len(flat_comments),
        'max_depth': max_depth,
        'depth_distribution': dict(depth_distribution),
        'deepest_comment': selected_comment,
        'subtree_stats': subtree_stats,
        'children_count': dict(children_count),
        'tie_info': tie_info
    }

def analyze_subtree(deepest_comment, all_comments, comments_by_id):
    """Analyze the subtree rooted at the deepest comment, counting all replies"""
    
    def find_all_descendants(root_author, all_comments, visited=None, max_depth=50):
        """Iteratively find all descendants (replies) of a comment author"""
        if visited is None:
            visited = set()
        
        # Prevent infinite recursion and circular references
        if root_author in visited or max_depth <= 0:
            return []
        
        visited.add(root_author)
        descendants = []
        
        # Use a queue for breadth-first traversal to avoid deep recursion
        queue = [root_author]
        current_depth = 0
        
        while queue and current_depth < max_depth:
            current_level = []
            next_queue = []
            
            for parent_author in queue:
                # Find direct children of this parent
                for comment in all_comments:
                    child_author = comment.get('author')
                    if (comment.get('parent_id') == parent_author and 
                        child_author not in visited and
                        child_author is not None):
                        
                        descendants.append(comment)
                        visited.add(child_author)
                        next_queue.append(child_author)
            
            queue = next_queue
            current_depth += 1
        
        return descendants
    
    # Get the root comment details
    root_author = deepest_comment.get('author')
    root_depth = deepest_comment.get('depth', deepest_comment.get('calculated_depth', 0))
    
    # Safety check
    if root_author is None:
        return {
            'total_comments': 0,
            'total_including_root': 1,
            'max_depth': root_depth,
            'root_comment_author': root_author,
            'root_absolute_depth': root_depth,
            'reply_count': 0
        }
    
    # Find all descendants (replies) in the subtree
    descendants = find_all_descendants(root_author, all_comments)
    
    # Calculate subtree statistics
    if descendants:
        # Only look at descendant depths for max subtree depth
        descendant_depths = [
            comment.get('depth', comment.get('calculated_depth', 0)) 
            for comment in descendants
        ]
        max_subtree_depth = max(descendant_depths) if descendant_depths else root_depth
        total_subtree_comments = 1 + len(descendants)  # Root + descendants
    else:
        max_subtree_depth = root_depth  # No replies = same as root depth
        total_subtree_comments = 1  # Just the root comment
    
    return {
        'total_comments': len(descendants),  # Number of replies (excluding the root)
        'total_including_root': total_subtree_comments,  # Total including root
        'max_depth': max_subtree_depth,  # Maximum depth in the subtree
        'root_comment_author': root_author,
        'root_absolute_depth': root_depth,
        'reply_count': len(descendants)  # Explicit count of replies
    }

def generate_research_table(conversations):
    """Generate a comprehensive research table"""
    
    # Separate conversations by type
    richly_branching = [conv for conv in conversations if conv['conversation_type'] == 'richly branching']
    poorly_branching = [conv for conv in conversations if conv['conversation_type'] == 'poorly branching']
    
    def calculate_metrics(conversations, category_name):
        """Calculate metrics for a category of conversations"""
        if not conversations:
            return {
                'category': category_name,
                'count': 0,
                'original_total_comments': 0,
                'original_mean_comments': 0,
                'original_total_depth': 0,
                'original_mean_depth': 0,
                'subtree_total_comments': 0,
                'subtree_mean_comments': 0,
                'subtree_total_depth': 0,
                'subtree_mean_depth': 0,
                'tie_statistics': {'posts_with_ties': 0, 'largest_subtree_wins': 0, 'first_found_fallbacks': 0}
            }
        
        original_comments = []
        original_depths = []
        subtree_comments = []
        subtree_depths = []
        tie_stats = {'posts_with_ties': 0, 'largest_subtree_wins': 0, 'first_found_fallbacks': 0}
        
        for conv in conversations:
            analysis = analyze_conversation_structure(conv['data'])
            
            # Original post metrics
            original_comments.append(analysis['total_comments'])
            original_depths.append(analysis['max_depth'])
            
            # Track tie information
            if analysis['tie_info'] and analysis['tie_info']['max_depth_comment_count'] > 1:
                tie_stats['posts_with_ties'] += 1
                
                if analysis['tie_info']['tie_breaking_used'] == 'largest_subtree':
                    tie_stats['largest_subtree_wins'] += 1
                elif analysis['tie_info']['tie_breaking_used'] == 'largest_subtree_then_first_found':
                    tie_stats['first_found_fallbacks'] += 1
            
            # Subtree metrics - using reply_count and properly calculated subtree max depth
            if analysis['subtree_stats']:
                subtree_comments.append(analysis['subtree_stats']['reply_count'])
                subtree_depths.append(analysis['subtree_stats']['max_depth'])
            else:
                subtree_comments.append(0)
                # If no subtree stats, the subtree max depth is just the original max depth
                subtree_depths.append(analysis['max_depth'])
        
        return {
            'category': category_name,
            'count': len(conversations),
            'original_total_comments': sum(original_comments),
            'original_mean_comments': np.mean(original_comments) if original_comments else 0,
            'original_max_depth': max(original_depths) if original_depths else 0,
            'original_mean_depth': np.mean(original_depths) if original_depths else 0,
            'subtree_total_comments': sum(subtree_comments),
            'subtree_mean_comments': np.mean(subtree_comments) if subtree_comments else 0,
            'subtree_max_depth': max(subtree_depths) if subtree_depths else 0,
            'subtree_mean_depth': np.mean(subtree_depths) if subtree_depths else 0,
            'tie_statistics': tie_stats
        }
    
    # Calculate metrics for both categories
    richly_metrics = calculate_metrics(richly_branching, "Richly Branching")
    poorly_metrics = calculate_metrics(poorly_branching, "Poorly Branching")
    
    # Print the research table
    print("\n" + "="*100)
    print("REDDIT CONVERSATION ANALYSIS - RESEARCH TABLE")
    print("Tie-breaking: Largest Subtree → First Found")
    print("="*100)
    
    header = f"{'Category':<18} {'Count':<6} {'Orig Total':<12} {'Orig Mean':<12} {'Orig Max D':<12} {'Orig D.Mean':<12} {'Sub Replies':<11} {'Sub Max D':<11} {'Sub D.Mean':<11}"
    print(header)
    print("-" * 90)
    
    def print_metrics(metrics):
        print(f"{metrics['category']:<18} "
              f"{metrics['count']:<6} "
              f"{metrics['original_total_comments']:<12} "
              f"{metrics['original_mean_comments']:<12.1f} "
              f"{metrics['original_max_depth']:<12} "
              f"{metrics['original_mean_depth']:<12.1f} "
              f"{metrics['subtree_total_comments']:<11} "
              f"{metrics['subtree_max_depth']:<11} "
              f"{metrics['subtree_mean_depth']:<11.1f}")
    
    print_metrics(richly_metrics)
    print_metrics(poorly_metrics)
    
    print("-" * 100)
    
    # Display tie statistics
    print("\nTIE-BREAKING STATISTICS:")
    for category, metrics in [("Richly Branching", richly_metrics), ("Poorly Branching", poorly_metrics)]:
        ties = metrics['tie_statistics']['posts_with_ties']
        largest_wins = metrics['tie_statistics']['largest_subtree_wins'] 
        fallbacks = metrics['tie_statistics']['first_found_fallbacks']
        total = metrics['count']
        
        print(f"{category}:")
        print(f"  Posts with max-depth ties: {ties}/{total} ({ties/total*100 if total > 0 else 0:.1f}%)")
        if ties > 0:
            print(f"  Resolved by largest subtree: {largest_wins}")
            print(f"  Required first-found fallback: {fallbacks}")
    
    # Calculate and display differences
    if richly_metrics['count'] > 0 and poorly_metrics['count'] > 0:
        print("\nKEY COMPARISONS:")
        print(f"Comments per post: Richly ({richly_metrics['original_mean_comments']:.1f}) vs Poorly ({poorly_metrics['original_mean_comments']:.1f})")
        print(f"Mean depth: Richly ({richly_metrics['original_mean_depth']:.1f}) vs Poorly ({poorly_metrics['original_mean_depth']:.1f})")
        print(f"Subtree replies: Richly ({richly_metrics['subtree_mean_comments']:.1f}) vs Poorly ({poorly_metrics['subtree_mean_comments']:.1f})")
        print(f"Subtree depth: Richly ({richly_metrics['subtree_mean_depth']:.1f}) vs Poorly ({poorly_metrics['subtree_mean_depth']:.1f})")
    
    return richly_metrics, poorly_metrics

def main():
    """Main function to run the analysis"""
    folder_path = "json_data"  # Update this path as needed
    
    print("Loading conversations...")
    conversations = load_conversations_from_folder(folder_path)
    
    if not conversations:
        print("No conversations loaded. Please check the folder path and file format.")
        return
    
    print(f"\nAnalyzing {len(conversations)} conversations...")
    print("Using tie-breaking strategy: Largest Subtree → First Found")
    
    richly_metrics, poorly_metrics = generate_research_table(conversations)
    
    print(f"\nAnalysis complete!")
    print(f"Legend: Orig = Original Post, Sub = Deepest Subtree, Max D = Maximum Depth, D.Mean = Depth Mean")
    print(f"Sub Replies = Number of replies in the deepest comment's subtree")
    print(f"\nTie-breaking explanation:")
    print(f"• When multiple comments reach max depth, select the one with the largest subtree")
    print(f"• If subtree sizes are also tied, use the first one found")

if __name__ == "__main__":
    main()