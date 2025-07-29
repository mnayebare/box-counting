import json
import numpy as np
from datetime import datetime
import os
import csv
from collections import defaultdict

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

def analyze_deletion_type(comment):
    """
    Analyze the type of deletion for a comment
    
    Parameters:
    - comment: Comment dictionary from JSON data
    
    Returns:
    - Dictionary with deletion type information
    """
    body = comment.get('body', '').strip()
    author = comment.get('author', '').strip()
    
    # Check for different deletion patterns
    author_deleted = author.lower() in ['deleted', '[deleted]']
    body_deleted = body.lower() in ['[deleted]', '[removed]']
    
    deletion_type = {
        'is_deleted': False,
        'author_deleted': author_deleted,
        'body_deleted': body_deleted,
        'deletion_category': 'active'
    }
    
    if author_deleted and body_deleted:
        deletion_type.update({
            'is_deleted': True,
            'deletion_category': 'both_deleted'  # Both account and content deleted
        })
    elif author_deleted and not body_deleted:
        deletion_type.update({
            'is_deleted': True,
            'deletion_category': 'author_only'  # Account deleted, content preserved
        })
    elif not author_deleted and body_deleted:
        deletion_type.update({
            'is_deleted': True,
            'deletion_category': 'content_only'  # Content deleted, account exists
        })
    
    return deletion_type

def filter_deleted_content_comments(data):
    """
    Remove comments where content is deleted (Content Only and Both Deleted types)
    while preserving conversation tree structure by reconnecting orphaned replies
    PROCESSES THE ENTIRE TREE - NO DEPTH CAPPING
    
    Parameters:
    - data: Original conversation data dictionary
    
    Returns:
    - Filtered conversation data with content-deleted comments removed (full depth)
    """
    def should_keep_comment(comment):
        """Check if a comment should be kept (content is not deleted)"""
        deletion_info = analyze_deletion_type(comment)
        # Keep only: 'active' and 'author_only' (where content is preserved)
        return deletion_info['deletion_category'] in ['active', 'author_only']
    
    def filter_and_reconnect(comments, current_depth=0):
        """
        Recursively filter comments and reconnect orphaned replies
        NO DEPTH LIMITS - processes entire tree
        """
        filtered_comments = []
        
        for comment in comments:
            if should_keep_comment(comment):
                # Keep this comment
                filtered_comment = comment.copy()
                filtered_comment['depth'] = current_depth
                
                # Process its replies (ALL DEPTHS)
                if comment.get('replies'):
                    filtered_replies = filter_and_reconnect(comment['replies'], current_depth + 1)
                    filtered_comment['replies'] = filtered_replies
                else:
                    filtered_comment['replies'] = []
                
                filtered_comments.append(filtered_comment)
                
            else:
                # Skip this comment but rescue its replies (reconnect orphans)
                if comment.get('replies'):
                    # Move replies up one level and adjust their depth
                    rescued_replies = filter_and_reconnect(comment['replies'], current_depth)
                    filtered_comments.extend(rescued_replies)
        
        return filtered_comments
    
    # Create a copy of the original data
    filtered_data = data.copy()
    
    # Filter comments (ENTIRE TREE - NO DEPTH CAPPING)
    if 'comments' in filtered_data:
        filtered_data['comments'] = filter_and_reconnect(filtered_data['comments'], 0)
    
    # Update title to indicate filtering
    original_title = filtered_data.get('post_title', 'Unknown')
    filtered_data['post_title'] = f"{original_title} (Content-Filtered - Full Tree)"
    
    return filtered_data

def analyze_filtering_impact(original_data, filtered_data):
    """
    Analyze the impact of content filtering on conversation structure
    
    Returns statistics about what was removed
    """
    def count_comments_recursive(comments):
        total = len(comments)
        for comment in comments:
            if comment.get('replies'):
                total += count_comments_recursive(comment['replies'])
        return total
    
    original_count = count_comments_recursive(original_data.get('comments', []))
    filtered_count = count_comments_recursive(filtered_data.get('comments', []))
    removed_count = original_count - filtered_count
    removal_percentage = (removed_count / original_count * 100) if original_count > 0 else 0
    
    original_max_depth = get_max_depth_from_data(original_data)
    filtered_max_depth = get_max_depth_from_data(filtered_data)
    
    return {
        'original_count': original_count,
        'filtered_count': filtered_count,
        'removed_count': removed_count,
        'removal_percentage': removal_percentage,
        'original_max_depth': original_max_depth,
        'filtered_max_depth': filtered_max_depth
    }

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

def analyze_deleted_comments(conversation_data):
    """
    Analyze deleted comments in a conversation with detailed categorization
    
    Parameters:
    - conversation_data: The conversation data dictionary
    
    Returns:
    - Dictionary with deletion statistics
    """
    total_comments = 0
    deletion_counts = {
        'both_deleted': 0,      # "author": "Deleted" AND "body": "[deleted]"
        'author_only': 0,       # "author": "Deleted" but content preserved
        'content_only': 0,      # "body": "[deleted]" but author exists
        'active': 0             # Neither deleted
    }
    
    deleted_by_depth = defaultdict(lambda: {'both_deleted': 0, 'author_only': 0, 'content_only': 0, 'active': 0})
    total_by_depth = defaultdict(int)
    deletion_details = []
    
    def traverse_comments(comments, parent_id="root"):
        nonlocal total_comments
        
        for i, comment in enumerate(comments):
            total_comments += 1
            current_depth = comment.get('depth', 0)
            total_by_depth[current_depth] += 1
            
            # Analyze deletion type
            deletion_info = analyze_deletion_type(comment)
            category = deletion_info['deletion_category']
            
            # Count by category
            deletion_counts[category] += 1
            deleted_by_depth[current_depth][category] += 1
            
            # Store deletion details for deleted comments
            if deletion_info['is_deleted']:
                deletion_details.append({
                    'comment_id': comment.get('id', f'unknown_{i}'),
                    'depth': current_depth,
                    'author': comment.get('author', 'Unknown'),
                    'body': comment.get('body', ''),
                    'score': comment.get('score', 0),
                    'timestamp': comment.get('timestamp', 'Unknown'),
                    'parent_id': parent_id,
                    'deletion_type': category,
                    'author_deleted': deletion_info['author_deleted'],
                    'body_deleted': deletion_info['body_deleted']
                })
            
            # Recursively process replies
            if comment.get('replies'):
                traverse_comments(comment['replies'], comment.get('id', f'{parent_id}_{i}'))
    
    # Start analysis
    traverse_comments(conversation_data.get('comments', []))
    
    # Calculate totals and percentages
    total_deleted = deletion_counts['both_deleted'] + deletion_counts['author_only'] + deletion_counts['content_only']
    deletion_percentage = (total_deleted / total_comments * 100) if total_comments > 0 else 0
    
    # Calculate deletion rate by depth and type
    deletion_rate_by_depth = {}
    for depth in total_by_depth:
        if total_by_depth[depth] > 0:
            depth_total_deleted = (deleted_by_depth[depth]['both_deleted'] + 
                                 deleted_by_depth[depth]['author_only'] + 
                                 deleted_by_depth[depth]['content_only'])
            deletion_rate_by_depth[depth] = (depth_total_deleted / total_by_depth[depth]) * 100
        else:
            deletion_rate_by_depth[depth] = 0
    
    return {
        'total_comments': total_comments,
        'deletion_counts': deletion_counts,
        'total_deleted': total_deleted,
        'deletion_percentage': deletion_percentage,
        'deleted_by_depth': dict(deleted_by_depth),
        'total_by_depth': dict(total_by_depth),
        'deletion_rate_by_depth': deletion_rate_by_depth,
        'deletion_details': deletion_details,
        'max_depth': max(total_by_depth.keys()) if total_by_depth else 0
    }

def print_before_after_summary_table(all_results):
    """
    Print before and after comparison tables showing the impact of content filtering
    
    Parameters:
    - all_results: List of deletion analysis results
    """
    if not all_results:
        print("No results to display in table")
        return
    
    # Separate by conversation type
    richly_branching = [r for r in all_results if r['conversation_type'] == 'richly branching']
    poorly_branching = [r for r in all_results if r['conversation_type'] == 'poorly branching']
    
    def calculate_before_stats(conversations):
        """Calculate statistics BEFORE filtering"""
        if not conversations:
            return {
                'total_comments': 0,
                'removed_comments': 0,
                'max_depth': 0,
                'count': 0
            }
        
        return {
            'total_comments': sum(r['original_count'] for r in conversations),
            'removed_comments': sum(r['removed_count'] for r in conversations),
            'max_depth': max(r['original_max_depth'] for r in conversations) if conversations else 0,
            'count': len(conversations)
        }
    
    def calculate_after_stats(conversations):
        """Calculate statistics AFTER filtering"""
        if not conversations:
            return {
                'total_comments': 0,
                'both_deleted': 0,
                'author_only': 0,
                'content_only': 0,
                'active': 0,
                'max_depth': 0,
                'count': 0
            }
        
        return {
            'total_comments': sum(r['analysis']['total_comments'] for r in conversations),
            'both_deleted': sum(r['analysis']['deletion_counts']['both_deleted'] for r in conversations),
            'author_only': sum(r['analysis']['deletion_counts']['author_only'] for r in conversations),
            'content_only': sum(r['analysis']['deletion_counts']['content_only'] for r in conversations),
            'active': sum(r['analysis']['deletion_counts']['active'] for r in conversations),
            'max_depth': max(r['analysis']['max_depth'] for r in conversations) if conversations else 0,
            'count': len(conversations)
        }
    
    rb_before = calculate_before_stats(richly_branching)
    pb_before = calculate_before_stats(poorly_branching)
    rb_after = calculate_after_stats(richly_branching)
    pb_after = calculate_after_stats(poorly_branching)
    
    print(f"\n{'='*120}")
    print("BEFORE vs AFTER CONTENT FILTERING COMPARISON")
    print(f"{'='*120}")
    
    # BEFORE TABLE
    print(f"\n📊 BEFORE FILTERING (Original Data):")
    print(f"{'-'*80}")
    print(f"{'Conversation Type':<20} {'Count':<8} {'Total Comments':<15} {'Max Depth':<12} {'Comments to Remove':<18}")
    print(f"{'-'*80}")
    print(f"{'Richly Branching':<20} {rb_before['count']:<8} {rb_before['total_comments']:<15} {rb_before['max_depth']:<12} {rb_before['removed_comments']:<18}")
    print(f"{'Poorly Branching':<20} {pb_before['count']:<8} {pb_before['total_comments']:<15} {pb_before['max_depth']:<12} {pb_before['removed_comments']:<18}")
    
    # Calculate totals for before
    total_before_count = rb_before['count'] + pb_before['count']
    total_before_comments = rb_before['total_comments'] + pb_before['total_comments']
    total_before_depth = max(rb_before['max_depth'], pb_before['max_depth'])
    total_before_removed = rb_before['removed_comments'] + pb_before['removed_comments']
    
    print(f"{'-'*80}")
    print(f"{'TOTAL':<20} {total_before_count:<8} {total_before_comments:<15} {total_before_depth:<12} {total_before_removed:<18}")
    print(f"{'-'*80}")
    
    # AFTER TABLE
    print(f"\n📊 AFTER FILTERING (Clean Data):")
    print(f"{'-'*110}")
    print(f"{'Conversation Type':<20} {'Count':<8} {'Total':<8} {'Both Del':<10} {'Author Del':<12} {'Content Del':<12} {'Active':<8} {'Max Depth':<12}")
    print(f"{'-'*110}")
    print(f"{'Richly Branching':<20} {rb_after['count']:<8} {rb_after['total_comments']:<8} {rb_after['both_deleted']:<10} {rb_after['author_only']:<12} {rb_after['content_only']:<12} {rb_after['active']:<8} {rb_after['max_depth']:<12}")
    print(f"{'Poorly Branching':<20} {pb_after['count']:<8} {pb_after['total_comments']:<8} {pb_after['both_deleted']:<10} {pb_after['author_only']:<12} {pb_after['content_only']:<12} {pb_after['active']:<8} {pb_after['max_depth']:<12}")
    
    # Calculate totals for after
    total_after_count = rb_after['count'] + pb_after['count']
    total_after_comments = rb_after['total_comments'] + pb_after['total_comments']
    total_after_both = rb_after['both_deleted'] + pb_after['both_deleted']
    total_after_author = rb_after['author_only'] + pb_after['author_only']
    total_after_content = rb_after['content_only'] + pb_after['content_only']
    total_after_active = rb_after['active'] + pb_after['active']
    total_after_depth = max(rb_after['max_depth'], pb_after['max_depth'])
    
    print(f"{'-'*110}")
    print(f"{'TOTAL':<20} {total_after_count:<8} {total_after_comments:<8} {total_after_both:<10} {total_after_author:<12} {total_after_content:<12} {total_after_active:<8} {total_after_depth:<12}")
    print(f"{'-'*110}")
    
    # IMPACT SUMMARY
    print(f"\n📈 FILTERING IMPACT SUMMARY:")
    print(f"{'-'*60}")
    
    if total_before_comments > 0:
        removal_rate = (total_before_removed / total_before_comments) * 100
        print(f"  Original total comments: {total_before_comments:,}")
        print(f"  Comments removed: {total_before_removed:,} ({removal_rate:.1f}%)")
        print(f"  Comments remaining: {total_after_comments:,} ({(total_after_comments/total_before_comments)*100:.1f}%)")
        print(f"  Depth preservation: {total_before_depth} → {total_after_depth}")
        
        print(f"\n  RICHLY BRANCHING:")
        if rb_before['total_comments'] > 0:
            rb_removal_rate = (rb_before['removed_comments'] / rb_before['total_comments']) * 100
            print(f"    Removed: {rb_before['removed_comments']:,}/{rb_before['total_comments']:,} ({rb_removal_rate:.1f}%)")
            print(f"    Remaining: {rb_after['total_comments']:,} ({(rb_after['total_comments']/rb_before['total_comments'])*100:.1f}%)")
        
        print(f"\n  POORLY BRANCHING:")
        if pb_before['total_comments'] > 0:
            pb_removal_rate = (pb_before['removed_comments'] / pb_before['total_comments']) * 100
            print(f"    Removed: {pb_before['removed_comments']:,}/{pb_before['total_comments']:,} ({pb_removal_rate:.1f}%)")
            print(f"    Remaining: {pb_after['total_comments']:,} ({(pb_after['total_comments']/pb_before['total_comments'])*100:.1f}%)")
    
    print(f"\n✅ VALIDATION:")
    print(f"  Both Deleted: {total_after_both} (should be 0)")
    print(f"  Content Only: {total_after_content} (should be 0)")
    print(f"  Author Only: {total_after_author} (content preserved from deleted accounts)")
    print(f"  Active Comments: {total_after_active} (clean, usable data)")
    
    print(f"\n{'='*120}")

def print_deletion_summary(all_results):
    """
    Print a comprehensive summary of deletion analysis with detailed categorization
    
    Parameters:
    - all_results: List of deletion analysis results
    """
    if not all_results:
        print("No results to summarize")
        return
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE CONTENT FILTERING SUMMARY (FULL DEPTH)")
    print(f"{'='*80}")
    
    # Overall statistics
    total_conversations = len(all_results)
    total_comments_all = sum(r['analysis']['total_comments'] for r in all_results)
    
    # Show original vs filtered counts
    original_counts = [r['original_count'] for r in all_results]
    filtered_counts = [r['filtered_count'] for r in all_results]
    removed_counts = [r['removed_count'] for r in all_results]
    
    print(f"Total conversations processed: {total_conversations}")
    print(f"📊 CONTENT FILTERING SUMMARY:")
    print(f"   Original total comments: {sum(original_counts):,}")
    print(f"   After filtering: {sum(filtered_counts):,}")
    print(f"   Total removed: {sum(removed_counts):,} ({sum(removed_counts)/sum(original_counts)*100:.1f}%)")
    print(f"   Comments in analysis: {total_comments_all:,} (full depth preserved)")
    
    # Aggregate deletion counts by type (should mostly be Author Only now)
    total_both_deleted = sum(r['analysis']['deletion_counts']['both_deleted'] for r in all_results)
    total_author_only = sum(r['analysis']['deletion_counts']['author_only'] for r in all_results)
    total_content_only = sum(r['analysis']['deletion_counts']['content_only'] for r in all_results)
    total_active = sum(r['analysis']['deletion_counts']['active'] for r in all_results)
    
    total_deleted_all = total_both_deleted + total_author_only + total_content_only
    overall_deletion_rate = (total_deleted_all / total_comments_all * 100) if total_comments_all > 0 else 0
    
    print(f"\n--- REMAINING DELETION TYPES (AFTER CONTENT FILTERING) ---")
    print(f"📍 BOTH DELETED: {total_both_deleted:,} (should be 0 - {total_both_deleted/total_comments_all*100:.2f}%)")
    print(f"👤 AUTHOR ONLY (preserved content): {total_author_only:,} ({total_author_only/total_comments_all*100:.2f}%)")
    print(f"💬 CONTENT ONLY: {total_content_only:,} (should be 0 - {total_content_only/total_comments_all*100:.2f}%)")
    print(f"✅ ACTIVE COMMENTS: {total_active:,} ({total_active/total_comments_all*100:.2f}%)")
    print(f"\n🔄 TOTAL REMAINING DELETIONS: {total_deleted_all:,} ({overall_deletion_rate:.2f}%)")
    
    # By conversation type
    richly_branching = [r for r in all_results if r['conversation_type'] == 'richly branching']
    poorly_branching = [r for r in all_results if r['conversation_type'] == 'poorly branching']
    
    print(f"\n{'='*60}")
    print("FILTERING IMPACT BY CONVERSATION TYPE")
    print(f"{'='*60}")
    
    def print_type_stats(conversations, type_name):
        if not conversations:
            return
            
        type_original = sum(r['original_count'] for r in conversations)
        type_filtered = sum(r['filtered_count'] for r in conversations)
        type_removed = sum(r['removed_count'] for r in conversations)
        type_removal_rate = (type_removed / type_original * 100) if type_original > 0 else 0
        
        type_both = sum(r['analysis']['deletion_counts']['both_deleted'] for r in conversations)
        type_author = sum(r['analysis']['deletion_counts']['author_only'] for r in conversations)
        type_content = sum(r['analysis']['deletion_counts']['content_only'] for r in conversations)
        type_active = sum(r['analysis']['deletion_counts']['active'] for r in conversations)
        
        print(f"\n🌳 {type_name.upper()} ({len(conversations)} conversations):")
        print(f"   Content filtering: {type_original:,} → {type_filtered:,} (removed {type_removed:,}, {type_removal_rate:.1f}%)")
        print(f"   📍 Both deleted: {type_both:,} | 👤 Author only: {type_author:,}")
        print(f"   💬 Content only: {type_content:,} | ✅ Active: {type_active:,}")
    
    print_type_stats(richly_branching, "Richly Branching")
    print_type_stats(poorly_branching, "Poorly Branching")
    
    # Per-conversation detailed breakdown
    print(f"\n{'='*80}")
    print("DETAILED PER-CONVERSATION FILTERING RESULTS")
    print(f"{'='*80}")
    
    for result in all_results:
        counts = result['analysis']['deletion_counts']
        
        print(f"\n📁 {result['post_id']} ({result['conversation_type']})")
        print(f"   Filtering: {result['original_count']} → {result['filtered_count']} (removed {result['removed_count']}, {result['removal_percentage']:.1f}%)")
        print(f"   Depth: {result['original_max_depth']} → {result['filtered_max_depth']} (preserved)")
        print(f"   Remaining: 📍{counts['both_deleted']} | 👤{counts['author_only']} | 💬{counts['content_only']} | ✅{counts['active']}")
    
    print(f"\n{'='*60}")
    print("LEGEND")
    print(f"{'='*60}")
    print(f"📍 BOTH DELETED = Should be 0 after filtering")
    print(f"👤 AUTHOR ONLY = Account deleted but content preserved (kept)")
    print(f"💬 CONTENT ONLY = Should be 0 after filtering")
    print(f"✅ ACTIVE = Normal comments with content and author (kept)")
    print(f"🎯 CONTENT FILTERING = Removed Both Deleted + Content Only types")
    print(f"📏 FULL DEPTH = No level-4 capping applied (handled by other scripts)")

def save_deletion_analysis_to_csv(all_results, filename='reddit_deletion_analysis.csv'):
    """
    Save deletion analysis results to CSV with content filtering information (full depth)
    
    Parameters:
    - all_results: List of deletion analysis results
    - filename: Output CSV filename
    """
    if not all_results:
        print("No deletion analysis data to save")
        return
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'post_id', 'conversation_type', 'total_comments', 'total_deleted',
            'both_deleted', 'author_only_deleted', 'content_only_deleted', 'active_comments',
            'deletion_percentage', 'max_depth', 'thread_title', 
            'original_count', 'filtered_count', 'removed_count', 'removal_percentage',
            'original_max_depth', 'filtered_max_depth',
            'content_filtering_applied', 'depth_cap_applied'
        ]
        
        # Add dynamic depth columns based on actual max depth found
        max_depth_found = max(r['analysis']['max_depth'] for r in all_results) if all_results else 0
        for depth in range(min(max_depth_found + 1, 20)):  # Limit to 20 depths max
            fieldnames.extend([
                f'depth_{depth}_total', f'depth_{depth}_both', f'depth_{depth}_author', 
                f'depth_{depth}_content', f'depth_{depth}_active'
            ])
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in all_results:
            # Prepare row data
            row = {
                'post_id': result['post_id'],
                'conversation_type': result['conversation_type'],
                'total_comments': result['analysis']['total_comments'],
                'total_deleted': result['analysis']['total_deleted'],
                'both_deleted': result['analysis']['deletion_counts']['both_deleted'],
                'author_only_deleted': result['analysis']['deletion_counts']['author_only'],
                'content_only_deleted': result['analysis']['deletion_counts']['content_only'],
                'active_comments': result['analysis']['deletion_counts']['active'],
                'deletion_percentage': round(result['analysis']['deletion_percentage'], 2),
                'max_depth': result['analysis']['max_depth'],
                'thread_title': result['thread_title'],
                'original_count': result.get('original_count', 'Unknown'),
                'filtered_count': result.get('filtered_count', 'Unknown'),
                'removed_count': result.get('removed_count', 'Unknown'),
                'removal_percentage': round(result.get('removal_percentage', 0), 2),
                'original_max_depth': result.get('original_max_depth', 'Unknown'),
                'filtered_max_depth': result.get('filtered_max_depth', 'Unknown'),
                'content_filtering_applied': result.get('content_filtering_applied', 'Yes'),
                'depth_cap_applied': result.get('depth_cap_applied', 'No')
            }
            
            # Add depth-specific data (all depths found)
            for depth in range(min(max_depth_found + 1, 20)):
                total_key = f'depth_{depth}_total'
                both_key = f'depth_{depth}_both'
                author_key = f'depth_{depth}_author'
                content_key = f'depth_{depth}_content'
                active_key = f'depth_{depth}_active'
                
                row[total_key] = result['analysis']['total_by_depth'].get(depth, 0)
                
                depth_data = result['analysis']['deleted_by_depth'].get(depth, {})
                row[both_key] = depth_data.get('both_deleted', 0)
                row[author_key] = depth_data.get('author_only', 0)
                row[content_key] = depth_data.get('content_only', 0)
                row[active_key] = depth_data.get('active', 0)
            
            writer.writerow(row)
    
    print(f"Content-filtered deletion analysis saved to: {filename}")
    print(f"Total rows written: {len(all_results)}")
    max_depth_display = min(max_depth_found, 19) if all_results else 0
    print(f"Depth columns included: 0 to {max_depth_display} (full depth preserved)")

def save_deletion_details_to_csv(all_results, filename='reddit_deletion_details.csv'):
    """
    Save detailed information about each deleted comment to CSV
    
    Parameters:
    - all_results: List of deletion analysis results
    - filename: Output CSV filename
    """
    if not all_results:
        print("No deletion details to save")
        return
    
    # Flatten all deletion details
    all_details = []
    for result in all_results:
        post_id = result['post_id']
        conversation_type = result['conversation_type']
        
        for detail in result['analysis']['deletion_details']:
            detail_row = {
                'post_id': post_id,
                'conversation_type': conversation_type,
                'comment_id': detail['comment_id'],
                'depth': detail['depth'],
                'author': detail['author'],
                'body': detail['body'][:100] + '...' if len(detail['body']) > 100 else detail['body'],  # Truncate long bodies
                'score': detail['score'],
                'timestamp': detail['timestamp'],
                'parent_id': detail['parent_id'],
                'deletion_type': detail['deletion_type'],
                'author_deleted': detail['author_deleted'],
                'body_deleted': detail['body_deleted']
            }
            all_details.append(detail_row)
    
    if not all_details:
        print("No deleted comment details found")
        return
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['post_id', 'conversation_type', 'comment_id', 'depth', 'author', 
                     'body', 'score', 'timestamp', 'parent_id', 'deletion_type',
                     'author_deleted', 'body_deleted']
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for detail in all_details:
            writer.writerow(detail)
    
    print(f"Deletion details saved to: {filename}")
    print(f"Total deleted comments recorded: {len(all_details)}")

# MAIN EXECUTION
if __name__ == "__main__":
    print("Running Reddit Content Filtering Analysis...")
    print("Removing Content Only + Both Deleted comments while preserving full conversation depth")
    
    # Specify the folder containing your JSON files
    json_data_folder = "json_data"  # Change this path if your folder is named differently
    
    # Load all conversations from the folder
    print(f"\n=== Loading conversations from '{json_data_folder}' folder ===")
    conversations = load_conversations_from_folder(json_data_folder)
    
    if not conversations:
        print("No conversation files found. Please check the folder path and file naming.")
        exit(1)
    
    # Create output directory for results
    output_dir = "deletion_analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Store all results
    all_results = []
    
    # Process each conversation
    for i, conv in enumerate(conversations, 1):
        print(f"\n{'='*60}")
        print(f"Analyzing deletions in {i}/{len(conversations)}: {conv['post_id']} ({conv['conversation_type']}) - CONTENT FILTERING ONLY")
        print(f"{'='*60}")
        
        try:
            # Apply content filtering ONLY (remove Content Only + Both Deleted)
            # CLEAN THE ENTIRE TREE - NO DEPTH CAPPING
            original_data = conv['data']
            filtered_data = filter_deleted_content_comments(original_data)
            
            # Analyze filtering impact on FULL TREE
            filtering_impact = analyze_filtering_impact(original_data, filtered_data)
            
            print(f"Content filtering applied to FULL TREE:")
            print(f"  Original comments: {filtering_impact['original_count']}")
            print(f"  After filtering: {filtering_impact['filtered_count']}")
            print(f"  Removed: {filtering_impact['removed_count']} ({filtering_impact['removal_percentage']:.1f}%)")
            print(f"  Depth: {filtering_impact['original_max_depth']} → {filtering_impact['filtered_max_depth']}")
            
            # UPDATE CONVERSATION DATA IN-PLACE (no new files, no level-4 capping)
            conv['data'] = filtered_data  # Replace original data with filtered data only
            
            # Analyze deleted comments on the filtered data (full depth)
            deletion_analysis = analyze_deleted_comments(filtered_data)
            
            # Store results
            result = {
                'post_id': conv['post_id'],
                'conversation_type': conv['conversation_type'],
                'thread_title': filtered_data.get('post_title', 'Unknown'),
                'analysis': deletion_analysis,
                'original_count': filtering_impact['original_count'],
                'filtered_count': filtering_impact['filtered_count'],
                'removed_count': filtering_impact['removed_count'],
                'removal_percentage': filtering_impact['removal_percentage'],
                'original_max_depth': filtering_impact['original_max_depth'],
                'filtered_max_depth': filtering_impact['filtered_max_depth'],
                'content_filtering_applied': True,
                'depth_cap_applied': False  # No depth capping applied here
            }
            
            all_results.append(result)
            
            # Print individual results with detailed breakdown
            print(f"Final analysis (Content-Filtered ONLY - Full Depth):")
            print(f"📊 Total comments: {deletion_analysis['total_comments']}")
            print(f"📍 Both deleted: {deletion_analysis['deletion_counts']['both_deleted']} (should be 0)")
            print(f"👤 Author only deleted: {deletion_analysis['deletion_counts']['author_only']}")
            print(f"💬 Content only deleted: {deletion_analysis['deletion_counts']['content_only']} (should be 0)")
            print(f"✅ Active comments: {deletion_analysis['deletion_counts']['active']}")
            print(f"🔄 Total deleted: {deletion_analysis['total_deleted']} (should only be Author Only)")
            print(f"📏 Max depth preserved: {deletion_analysis['max_depth']}")
            
            if deletion_analysis['total_deleted'] > 0:
                print(f"Remaining deletions by depth: {deletion_analysis['deleted_by_depth']}")
                
        except Exception as e:
            print(f"Error analyzing {conv['post_id']}: {e}")
            continue
    
    # Generate before/after comparison table
    print_before_after_summary_table(all_results)
    
    # Generate comprehensive summary
    print_deletion_summary(all_results)
    
    # Save results to CSV files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    summary_csv = os.path.join(output_dir, f"content_filtered_deletion_analysis_{timestamp}.csv")
    details_csv = os.path.join(output_dir, f"content_filtered_deletion_details_{timestamp}.csv")
    
    save_deletion_analysis_to_csv(all_results, summary_csv)
    save_deletion_details_to_csv(all_results, details_csv)
    
    print(f"\n{'='*80}")
    print("CONTENT FILTERING COMPLETE")
    print(f"{'='*80}")
    print(f"Analysis results saved to:")
    print(f"  - Summary: {summary_csv}")
    print(f"  - Details: {details_csv}")
    print(f"\nTotal conversations processed: {len(all_results)}")
    print(f"Output directory: {output_dir}/")
    print(f"\n🎯 PROCESSING APPLIED:")
    print(f"   ✅ Content filtering on FULL TREE (removed Content Only + Both Deleted)")
    print(f"   ✅ Orphaned replies reconnected to maintain tree structure")
    print(f"   ✅ Original conversation data MODIFIED IN-PLACE")
    print(f"   ✅ All depths preserved (no level-4 capping)")
    print(f"\n💡 RESULT: Your JSON files now contain clean conversation data")
    print(f"🚀 READY: Run your fractal analysis scripts - they will handle level-4 capping")