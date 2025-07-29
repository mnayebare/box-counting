import json
import csv
import math
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from scipy import stats

# Create graphs folder if it doesn't exist
graphs = 'graphs'
if not os.path.exists(graphs):
    os.makedirs(graphs)
    print(f"Created folder: {graphs}")

# Set academic publication style
plt.rcParams.update({
    'font.size': 14,  # Increased from 12
    'font.family': 'Arial',
    'axes.linewidth': 1.2,
    'axes.labelsize': 16,  # Increased from 14
    'axes.titlesize': 18,  # Increased from 16
    'xtick.labelsize': 14,  # Increased from 12
    'ytick.labelsize': 14,  # Increased from 12
    'legend.fontsize': 14,  # Increased from 12
    'figure.titlesize': 20,  # Increased from 18
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8
})

class CommentNode:
    def __init__(self, author, body, timestamp=None):
        self.author = author
        self.body = body
        self.timestamp = timestamp
        self.replies = []

    def add_reply(self, reply):
        self.replies.append(reply)

def calculate_80_percent_engagement_time(data):
    """
    Calculate the time when 80% of replies were posted
    """
    try:
        post_timestamp = data.get('post_timestamp', '')
        if not post_timestamp:
            return None, None
            
        post_time = datetime.strptime(post_timestamp, '%Y-%m-%d %H:%M:%S')
        
        # Collect all reply timestamps
        def collect_timestamps(comments):
            timestamps = []
            for comment in comments:
                try:
                    timestamp = datetime.strptime(comment['timestamp'], '%Y-%m-%d %H:%M:%S')
                    timestamps.append(timestamp)
                    if comment.get('replies'):
                        timestamps.extend(collect_timestamps(comment['replies']))
                except (ValueError, KeyError):
                    continue
            return timestamps
        
        all_timestamps = collect_timestamps(data["comments"])
        
        if not all_timestamps:
            return None, None
            
        all_timestamps.sort()
        total_replies = len(all_timestamps)
        target_80_percent = int(total_replies * 0.8)
        
        if target_80_percent > 0 and target_80_percent <= len(all_timestamps):
            cutoff_time = all_timestamps[target_80_percent - 1]
            engagement_hours = (cutoff_time - post_time).total_seconds() / 3600
            return cutoff_time, engagement_hours
        else:
            return None, None
            
    except Exception as e:
        print(f"Error calculating 80% engagement time: {e}")
        return None, None

def build_conversation_tree_from_reddit(comments_data, post_time=None, cutoff_time=None):
    """ 
    Builds a tree from Reddit JSON format with nested replies
    """
    nodes = []
    
    def process_comment(comment_data):
        # Parse comment timestamp if available
        comment_time = None
        try:
            if 'timestamp' in comment_data:
                comment_time = datetime.strptime(comment_data["timestamp"], '%Y-%m-%d %H:%M:%S')
                # Skip comments outside time window if specified
                if cutoff_time and comment_time > cutoff_time:
                    return None
        except (ValueError, KeyError):
            pass
        
        # Create a node for this comment
        node = CommentNode(
            comment_data.get("author", "Unknown"), 
            comment_data.get("body", ""),
            comment_time
        )
        
        # Process replies recursively
        if "replies" in comment_data and comment_data["replies"]:
            for reply_data in comment_data["replies"]:
                reply_node = process_comment(reply_data)
                if reply_node is not None:
                    node.add_reply(reply_node)
        
        return node
    
    # Process all top-level comments
    for comment_data in comments_data:
        comment_node = process_comment(comment_data)
        if comment_node is not None:
            nodes.append(comment_node)
    
    return nodes

def compute_tree_weight(comment, scaling_factor=1.1):
    """ 
    Recursively computes the total weight of the entire subtree.
    Each node contributes (1.1^replies), and subtree weight = sum of all node weights
    """
    # This node's weight based on its direct replies
    node_weight = scaling_factor ** len(comment.replies)
    
    # Add weights from all child subtrees recursively
    child_weights = 0
    for reply in comment.replies:
        child_weights += compute_tree_weight(reply, scaling_factor)
    
    return node_weight + child_weights

def compute_node_weight(comment, scaling_factor=1.1):
    """ 
    Computes the weight of JUST THIS NODE: (1.1^replies)
    """
    return scaling_factor ** len(comment.replies)

def compute_subtree_max_depth(comment_node):
    """
    Computes the maximum depth of a subtree rooted at the given comment node.
    """
    if not comment_node.replies:
        return 0  # Leaf node has depth 0 from itself
    
    # Find the maximum depth among all child subtrees
    max_child_depth = max(compute_subtree_max_depth(reply) for reply in comment_node.replies)
    return max_child_depth + 1

def count_total_replies(comment_nodes):
    """
    Recursively counts all replies (nodes) in the conversation tree.
    """
    total = 0
    for node in comment_nodes:
        total += 1  # Count this node
        if node.replies:
            total += count_total_replies(node.replies)
    return total

def count_subtree_replies(comment_node):
    """
    Recursively counts all replies in a single subtree
    """
    count = 1  # Count this node
    for reply in comment_node.replies:
        count += count_subtree_replies(reply)
    return count

def analyze_conversation_structure(root_post_node, comment_nodes):
    """ 
    Analyzes tree structure with multiple metrics including subtree analysis
    """
    
    # Calculate tree weight
    tree_weight = compute_tree_weight(root_post_node)
    
    # Count total replies in the entire conversation
    total_replies = count_total_replies(comment_nodes)
    
    # Enhanced Subtree Analysis
    subtree_data = []
    
    for i, comment_node in enumerate(comment_nodes):
        subtree_depth = compute_subtree_max_depth(comment_node)
        subtree_weight = compute_tree_weight(comment_node)
        subtree_reply_count = count_subtree_replies(comment_node)
        
        subtree_info = {
            'index': i,
            'depth': subtree_depth,
            'weight': subtree_weight,
            'reply_count': subtree_reply_count,
            'node': comment_node
        }
        subtree_data.append(subtree_info)
    
    # Sort subtrees by weight (descending) to identify top performers
    subtree_data_sorted = sorted(subtree_data, key=lambda x: x['weight'], reverse=True)
    
    # Extract traditional metrics for compatibility
    subtree_depths = [s['depth'] for s in subtree_data]
    subtree_weights = [s['weight'] for s in subtree_data]
    
    # Calculate subtree statistics
    max_subtree_depth = max(subtree_depths) if subtree_depths else 0
    avg_subtree_depth = sum(subtree_depths) / len(subtree_depths) if subtree_depths else 0
    
    # Count subtrees by depth
    subtree_depth_counts = {}
    for depth in subtree_depths:
        subtree_depth_counts[depth] = subtree_depth_counts.get(depth, 0) + 1
    
    # Depth analysis
    depth_data = {}
    
    # Add root post at depth 0
    root_node_weight = compute_node_weight(root_post_node)
    depth_data[0] = {
        'nodes': [root_post_node],
        'node_weights': [root_node_weight],
        'node_count': 1,
        'depth_weight': root_node_weight
    }
    
    # Analyze comment tree starting at depth 1
    def collect_depth_info(node_list, depth):
        if depth not in depth_data:
            depth_data[depth] = {
                'nodes': [],
                'node_weights': [],
                'node_count': 0,
                'depth_weight': 0.0
            }
        
        for node in node_list:
            node_weight = compute_node_weight(node)
            depth_data[depth]['nodes'].append(node)
            depth_data[depth]['node_weights'].append(node_weight)
            depth_data[depth]['node_count'] += 1
            depth_data[depth]['depth_weight'] += node_weight
            
            # Recursively process replies at next depth level
            if node.replies:
                collect_depth_info(node.replies, depth + 1)
    
    # Start collecting comments at depth 1
    collect_depth_info(comment_nodes, 1)
    
    # Add subtree analysis to return data
    subtree_analysis = {
        'max_subtree_depth': max_subtree_depth,
        'avg_subtree_depth': avg_subtree_depth,
        'subtree_depth_counts': subtree_depth_counts,
        'total_subtrees': len(comment_nodes),
        'total_replies': total_replies,
        'subtree_depths': subtree_depths,
        'subtree_weights': subtree_weights,
        'subtree_data_sorted': subtree_data_sorted
    }
    
    return tree_weight, depth_data, subtree_analysis

def extract_tst_metrics(subtree_analysis, post_id, conversation_type, post_title, engagement_hours=None):
    """
    Extract Top Subtree (TST) metrics - enhanced with timing information
    """
    subtree_data_sorted = subtree_analysis['subtree_data_sorted']
    
    tst_metrics = []
    
    # Extract only the top subtree (TST)
    if subtree_data_sorted:  # Check if there are any subtrees
        subtree_info = subtree_data_sorted[0]  # Get the highest weight subtree
        
        # Get the top subtree's body as the title
        top_subtree_node = subtree_info['node']
        top_subtree_title = top_subtree_node.body
        
        # Truncate if too long (optional - adjust length as needed)
        if len(top_subtree_title) > 100:
            top_subtree_title = top_subtree_title[:97] + "..."
        
        # Calculate additional metrics
        total_subtrees = subtree_analysis['total_subtrees']
        weight_percentage = (subtree_info['weight'] / sum(subtree_analysis['subtree_weights'])) * 100 if subtree_analysis['subtree_weights'] else 0
        reply_percentage = (subtree_info['reply_count'] / subtree_analysis['total_replies']) * 100 if subtree_analysis['total_replies'] > 0 else 0
        
        # Analyze branching pattern of this top subtree (TST)
        node = subtree_info['node']
        direct_replies = len(node.replies)
        
        # Calculate branching efficiency (weight per direct reply)
        branching_efficiency = subtree_info['weight'] / direct_replies if direct_replies > 0 else subtree_info['weight']
        
        tst_metric = {
            'post_id': post_id,
            'conversation_type': conversation_type,
            'post_title': top_subtree_title,  # Top subtree body
            'original_post_title': post_title,    # Original post title for reference
            'tst_weight': subtree_info['weight'],
            'tst_depth': subtree_info['depth'],
            'tst_reply_count': subtree_info['reply_count'],
            'tst_direct_replies': direct_replies,
            'weight_percentage_of_total': weight_percentage,
            'reply_percentage_of_total': reply_percentage,
            'branching_efficiency': branching_efficiency,
            'subtree_index': subtree_info['index'],
            'total_subtrees_in_conversation': total_subtrees,
            'total_conversation_replies': subtree_analysis['total_replies'],
            'max_conversation_depth': subtree_analysis['max_subtree_depth'],
            'avg_conversation_depth': subtree_analysis['avg_subtree_depth'],
            'time_to_80_percent_engagement_hours': engagement_hours  # NEW: Add timing information
        }
        tst_metrics.append(tst_metric)
    
    return tst_metrics

def load_and_analyze_conversations():
    """
    Load conversations from JSON files and perform TST analysis with 80% engagement timing
    """
    all_tst_data = []
    timing_summary = []
    json_folder_path = 'json_data'
    
    try:
        if not os.path.exists(json_folder_path):
            print(f"Error: Folder '{json_folder_path}' not found.")
            return [], []
        
        json_files = [f for f in os.listdir(json_folder_path) 
                     if f.endswith('.json') and 'reddit_comments' in f]
        
        print(f"Found {len(json_files)} JSON files to process")
        
        for json_file in json_files:
            json_file_path = os.path.join(json_folder_path, json_file)
            
            try:
                with open(json_file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                
                # Extract post ID from file name
                post_id = json_file.replace('_reddit_comments_with_time.json', '').replace('.json', '')
                
                # Determine conversation type
                conversation_type = ""
                if "hb" in post_id.lower():
                    conversation_type = "richly branching"
                elif "lb" in post_id.lower():
                    conversation_type = "poorly branching"
                else:
                    conversation_type = "unknown"
                
                print(f"Processing: {post_id} ({conversation_type})")
                
                # Calculate 80% engagement timing
                cutoff_time, engagement_hours = calculate_80_percent_engagement_time(data)
                
                # Build conversation tree with 80% engagement filter
                if cutoff_time:
                    post_time = datetime.strptime(data.get('post_timestamp', ''), '%Y-%m-%d %H:%M:%S')
                    root_nodes = build_conversation_tree_from_reddit(data["comments"], post_time, cutoff_time)
                    print(f"  80% engagement reached at: {engagement_hours:.2f} hours")
                else:
                    # Fallback: process all comments
                    root_nodes = build_conversation_tree_from_reddit(data["comments"])
                    print(f"  No timing data available, processing all comments")
                
                # Create root post node
                root_post_node = CommentNode(
                    author=data.get('post_author', 'Unknown'),
                    body=data.get('post_title', 'No Title'),
                    timestamp=None
                )
                
                # Add all top-level comments as replies to the root post
                for comment_node in root_nodes:
                    root_post_node.add_reply(comment_node)
                
                # Analyze tree structure
                tree_weight, depth_data, subtree_analysis = analyze_conversation_structure(root_post_node, root_nodes)
                
                # Extract TST metrics with timing information
                tst_metrics = extract_tst_metrics(subtree_analysis, post_id, conversation_type, 
                                                data.get('post_title', 'Unknown'), engagement_hours)
                all_tst_data.extend(tst_metrics)
                
                # Store timing summary
                timing_summary.append({
                    'post_id': post_id,
                    'conversation_type': conversation_type,
                    'engagement_hours': engagement_hours,
                    'max_depth': subtree_analysis['max_subtree_depth'],
                    'total_replies': subtree_analysis['total_replies'],
                    'total_subtrees': subtree_analysis['total_subtrees']
                })
                
            except Exception as e:
                print(f"Error processing file {json_file}: {e}")
                continue
        
        return all_tst_data, timing_summary
        
    except Exception as e:
        print(f"Error accessing folder: {e}")
        return [], []

def create_enhanced_visualizations(tst_data, timing_summary):
    """
    Create enhanced visualizations - single panel for 80% engagement timing analysis
    """
    if not tst_data:
        print("No TST data available for visualization")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(tst_data)
    timing_df = pd.DataFrame(timing_summary)
    
    # Remove rows with missing timing data for timing analysis
    timing_df_clean = timing_df.dropna(subset=['engagement_hours'])
    
    # Separate by conversation type
    rb_timing = timing_df_clean[timing_df_clean['conversation_type'] == 'richly branching']
    pb_timing = timing_df_clean[timing_df_clean['conversation_type'] == 'poorly branching']
    
    # Academic colorblind-friendly colors
    colors = {
        'rich_branching': '#1f77b4',  # Blue
        'poor_branching': '#ff7f0e',  # Orange
        'sweet_spot': '#2ca02c'       # Green
    }
    
    # Set up the plotting with single panel - larger size for better visibility
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    fig.suptitle('80% Engagement Time Analysis', 
                 fontsize=24, fontweight='bold', y=0.95)
    
    # Calculate sample sizes for display
    n_rich_timing = len(rb_timing)
    n_poor_timing = len(pb_timing)
    
    # 80% Engagement Timing - LINE GRAPH with Sweet Spot
    if len(rb_timing) > 0 and len(pb_timing) > 0:
        # Create hourly time points (0 to 48 hours)
        time_points = np.arange(0, 49, 1)  # 0, 1, 2, 3, ... 48 hours
        
        # Calculate cumulative distribution for each conversation type
        rb_cumulative = []
        pb_cumulative = []
        
        for t in time_points:
            rb_pct = (rb_timing['engagement_hours'] <= t).sum() / len(rb_timing) * 100
            pb_pct = (pb_timing['engagement_hours'] <= t).sum() / len(pb_timing) * 100
            rb_cumulative.append(rb_pct)
            pb_cumulative.append(pb_pct)
        
        # Plot cumulative distribution lines with increased line width and marker size
        ax.plot(time_points, rb_cumulative, label=f'Rich Branching (n={n_rich_timing})', 
                color=colors['rich_branching'], linewidth=4, marker='o', markersize=5, 
                markevery=3, alpha=0.9)
        ax.plot(time_points, pb_cumulative, label=f'Poor Branching (n={n_poor_timing})', 
                color=colors['poor_branching'], linewidth=4, marker='s', markersize=5, 
                markevery=3, alpha=0.9)
        
        # Find sweet spot (where both types have at least 60% coverage)
        sweet_spot = None
        for i, t in enumerate(time_points):
            if rb_cumulative[i] >= 60 and pb_cumulative[i] >= 60:
                sweet_spot = t
                break
        
        # Add sweet spot line if found
        if sweet_spot is not None:
            ax.axvline(sweet_spot, color=colors['sweet_spot'], linestyle=':', 
                       linewidth=4, alpha=0.9, label=f'Sweet Spot: {sweet_spot}h')
            
            # Add sweet spot annotation with larger font
            rb_coverage_at_spot = rb_cumulative[sweet_spot] if sweet_spot < len(rb_cumulative) else rb_cumulative[-1]
            pb_coverage_at_spot = pb_cumulative[sweet_spot] if sweet_spot < len(pb_cumulative) else pb_cumulative[-1]
            
            ax.annotate(f'Sweet Spot\n{sweet_spot}h\nR:{rb_coverage_at_spot:.0f}% P:{pb_coverage_at_spot:.0f}%', 
                        xy=(sweet_spot, 70), xytext=(sweet_spot + 8, 50),
                        arrowprops=dict(arrowstyle='->', color=colors['sweet_spot'], lw=3),
                        fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', 
                                 edgecolor=colors['sweet_spot'], alpha=0.8))
        
        # Add median lines with increased line width
        rb_median = rb_timing['engagement_hours'].median()
        pb_median = pb_timing['engagement_hours'].median()
        
        ax.axvline(rb_median, color=colors['rich_branching'], linestyle='--', 
                   linewidth=3, alpha=0.8)
        ax.axvline(pb_median, color=colors['poor_branching'], linestyle='--', 
                   linewidth=3, alpha=0.8)
        
        # Add median labels with larger font
        ax.text(rb_median + 1, 25, f'Rich Median\n{rb_median:.1f}h', 
                color=colors['rich_branching'], fontweight='bold', fontsize=11)
        ax.text(pb_median + 1, 15, f'Poor Median\n{pb_median:.1f}h', 
                color=colors['poor_branching'], fontweight='bold', fontsize=11)
        
        ax.set_title('Cumulative 80% Engagement Time Distribution', fontweight='bold', pad=20, fontsize=20)
        ax.set_xlabel('Hours to 80% Engagement', fontweight='bold', fontsize=18)
        ax.set_ylabel('Conversations Captured (%)', fontweight='bold', fontsize=18)
        ax.set_ylim(0, 100)
        ax.set_xlim(0, 48)
        ax.legend(fontsize=14, loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # Store sweet spot for summary reporting
        globals()['calculated_sweet_spot'] = sweet_spot
        globals()['sweet_spot_coverage'] = {
            'rich': rb_coverage_at_spot if sweet_spot else None,
            'poor': pb_coverage_at_spot if sweet_spot else None
        }
        
    else:
        ax.text(0.5, 0.5, 'No timing data\navailable', ha='center', va='center', 
                transform=ax.transAxes, fontsize=16)
        ax.set_title('Cumulative 80% Engagement Time Distribution', fontweight='bold', pad=20, fontsize=20)
    
    # Improve overall layout with better spacing and padding
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.12, left=0.1, right=0.95)
    
    # Save the graph with academic standards and proper padding
    graph_filename = os.path.join(graphs, 'engagement_timing_analysis_single_panel.png')
    plt.savefig(graph_filename, dpi=300, bbox_inches='tight', facecolor='white', 
                pad_inches=0.3)
    graph_pdf = os.path.join(graphs, 'engagement_timing_analysis_single_panel.pdf')
    plt.savefig(graph_pdf, bbox_inches='tight', facecolor='white', 
                pad_inches=0.3)
    
    print(f"Saved engagement timing analysis graph: {graph_filename}")
    print(f"Saved engagement timing analysis graph (PDF): {graph_pdf}")
    plt.show()

def print_enhanced_summary(tst_data, timing_summary):
    """
    Print enhanced summary including timing analysis
    """
    if not tst_data:
        print("No data available for summary")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(tst_data)
    timing_df = pd.DataFrame(timing_summary)
    
    # Remove rows with missing timing data
    timing_df_clean = timing_df.dropna(subset=['engagement_hours'])
    
    # Separate by conversation type
    richly_branching = df[df['conversation_type'] == 'richly branching']
    poorly_branching = df[df['conversation_type'] == 'poorly branching']
    
    rb_timing = timing_df_clean[timing_df_clean['conversation_type'] == 'richly branching']
    pb_timing = timing_df_clean[timing_df_clean['conversation_type'] == 'poorly branching']
    
    print("\n" + "="*80)
    print("ENHANCED SUBTREE ANALYSIS WITH 80% ENGAGEMENT TIMING")
    print("="*80)
    print(f"Richly Branching Conversations: {len(richly_branching)}")
    print(f"Poorly Branching Conversations: {len(poorly_branching)}")
    print(f"Conversations with timing data: {len(timing_df_clean)} / {len(timing_df)}")
    
    # Analysis metrics
    metrics = [
        ('max_conversation_depth', 'Max Subtree Depth'),
        ('total_subtrees_in_conversation', 'Number of Subtrees'),
        ('total_conversation_replies', 'Total Replies'),
        ('tst_weight', 'TST Weight'),
        ('tst_depth', 'TST Depth')
    ]
    
    for metric, title in metrics:
        if metric in richly_branching.columns and metric in poorly_branching.columns:
            print(f"\n{title.upper()}:")
            print("-" * len(title))
            
            rich_mean = richly_branching[metric].mean()
            rich_std = richly_branching[metric].std()
            poor_mean = poorly_branching[metric].mean()
            poor_std = poorly_branching[metric].std()
            
            try:
                t_stat, p_val = stats.ttest_ind(richly_branching[metric], poorly_branching[metric])
            except:
                t_stat, p_val = 0, 1.0
            
            print(f"Richly Branching: Mean={rich_mean:.3f}, Std={rich_std:.3f}")
            print(f"Poorly Branching: Mean={poor_mean:.3f}, Std={poor_std:.3f}")
            print(f"Difference: {rich_mean - poor_mean:.3f} ({((rich_mean - poor_mean)/poor_mean*100):+.1f}%)")
            print(f"Statistical test: t={t_stat:.3f}, p={p_val:.3f}")
    
    # Timing analysis
    if len(rb_timing) > 0 and len(pb_timing) > 0:
        print(f"\n80% ENGAGEMENT TIMING ANALYSIS:")
        print("-" * 30)
        
        rb_time_mean = rb_timing['engagement_hours'].mean()
        rb_time_std = rb_timing['engagement_hours'].std()
        pb_time_mean = pb_timing['engagement_hours'].mean()
        pb_time_std = pb_timing['engagement_hours'].std()
        
        try:
            t_stat, p_val = stats.ttest_ind(rb_timing['engagement_hours'], pb_timing['engagement_hours'])
        except:
            t_stat, p_val = 0, 1.0
        
        print(f"Richly Branching (n={len(rb_timing)}): Mean={rb_time_mean:.2f}h, Std={rb_time_std:.2f}h")
        print(f"Poorly Branching (n={len(pb_timing)}): Mean={pb_time_mean:.2f}h, Std={pb_time_std:.2f}h")
        print(f"Difference: {rb_time_mean - pb_time_mean:.2f}h ({((rb_time_mean - pb_time_mean)/pb_time_mean*100):+.1f}%)")
        print(f"Statistical test: t={t_stat:.3f}, p={p_val:.3f}")
        
        # Timing ranges
        print(f"\nTiming Ranges:")
        print(f"Richly Branching: {rb_timing['engagement_hours'].min():.1f}h - {rb_timing['engagement_hours'].max():.1f}h")
        print(f"Poorly Branching: {pb_timing['engagement_hours'].min():.1f}h - {pb_timing['engagement_hours'].max():.1f}h")
        
        # Sweet spot analysis
        if 'calculated_sweet_spot' in globals() and globals()['calculated_sweet_spot'] is not None:
            sweet_spot = globals()['calculated_sweet_spot']
            coverage = globals()['sweet_spot_coverage']
            print(f"\n🎯 OPTIMAL DATA COLLECTION WINDOW: {sweet_spot} HOURS")
            print(f"At {sweet_spot}h, captures:")
            print(f"  • {coverage['rich']:.0f}% of rich branching conversations")
            print(f"  • {coverage['poor']:.0f}% of poor branching conversations")
            print(f"  • Balanced coverage for both conversation types")
        else:
            print(f"\n🎯 SWEET SPOT: Not found within 48-hour window")
            print("Consider extending analysis period or lowering coverage threshold")
        
        # Enhanced coverage analysis at key time points with depth information
        print(f"\nENHANCED COVERAGE ANALYSIS:")
        key_times = [12, 18, 24, 36, 48]
        print(f"{'Time':<6} {'Rich (%)':<10} {'Poor (%)':<10} {'Min (%)':<8} {'Rich Max Depth':<15} {'Poor Max Depth':<15} {'Max Diff':<10}")
        print("-" * 85)
        
        for t in key_times:
            # Time-based coverage
            rb_cov = 100 * (rb_timing['engagement_hours'] <= t).sum() / len(rb_timing)
            pb_cov = 100 * (pb_timing['engagement_hours'] <= t).sum() / len(pb_timing)
            min_cov = min(rb_cov, pb_cov)
            
            # Depth analysis for conversations captured by time t
            rb_conversations_by_t = rb_timing[rb_timing['engagement_hours'] <= t]
            pb_conversations_by_t = pb_timing[pb_timing['engagement_hours'] <= t]
            
            if len(rb_conversations_by_t) > 0:
                rb_max_depth = rb_conversations_by_t['max_depth'].max()
            else:
                rb_max_depth = 0
                
            if len(pb_conversations_by_t) > 0:
                pb_max_depth = pb_conversations_by_t['max_depth'].max()
            else:
                pb_max_depth = 0
            
            max_diff = rb_max_depth - pb_max_depth
            
            print(f"{t:2d}h    {rb_cov:6.1f}     {pb_cov:6.1f}     {min_cov:5.1f}    {rb_max_depth:11.0f}     {pb_max_depth:11.0f}     {max_diff:+6.0f}")
        
        # Additional depth-focused analysis
        print(f"\nDEPTH DISTRIBUTION BY TIME WINDOW:")
        print(f"{'Time Window':<12} {'Rich: Shallow':<15} {'Rich: Deep':<12} {'Poor: Shallow':<15} {'Poor: Deep':<12}")
        print(f"{'':>12} {'(≤6 levels)':<15} {'(>6 levels)':<12} {'(≤6 levels)':<15} {'(>6 levels)':<12}")
        print("-" * 80)
        
        for t in [18, 24, 36, 48]:
            rb_conversations_by_t = rb_timing[rb_timing['engagement_hours'] <= t]
            pb_conversations_by_t = pb_timing[pb_timing['engagement_hours'] <= t]
            
            if len(rb_conversations_by_t) > 0:
                rb_shallow = (rb_conversations_by_t['max_depth'] <= 6).sum()
                rb_deep = (rb_conversations_by_t['max_depth'] > 6).sum()
                rb_shallow_pct = rb_shallow / len(rb_conversations_by_t) * 100
                rb_deep_pct = rb_deep / len(rb_conversations_by_t) * 100
            else:
                rb_shallow_pct = rb_deep_pct = 0
                
            if len(pb_conversations_by_t) > 0:
                pb_shallow = (pb_conversations_by_t['max_depth'] <= 6).sum()
                pb_deep = (pb_conversations_by_t['max_depth'] > 6).sum()
                pb_shallow_pct = pb_shallow / len(pb_conversations_by_t) * 100
                pb_deep_pct = pb_deep / len(pb_conversations_by_t) * 100
            else:
                pb_shallow_pct = pb_deep_pct = 0
            
            print(f"By {t:2d}h       {rb_shallow_pct:6.1f}%         {rb_deep_pct:6.1f}%       {pb_shallow_pct:6.1f}%         {pb_deep_pct:6.1f}%")
        
        # Timing vs Depth insights without averages
        print(f"\nTIMING vs DEPTH INSIGHTS:")
        print("-" * 25)
        
        # Fast vs slow conversations depth analysis
        rb_fast = rb_timing[rb_timing['engagement_hours'] <= rb_timing['engagement_hours'].median()]
        rb_slow = rb_timing[rb_timing['engagement_hours'] > rb_timing['engagement_hours'].median()]
        pb_fast = pb_timing[pb_timing['engagement_hours'] <= pb_timing['engagement_hours'].median()]
        pb_slow = pb_timing[pb_timing['engagement_hours'] > pb_timing['engagement_hours'].median()]
        
        print(f"Rich Branching:")
        if len(rb_fast) > 0 and len(rb_slow) > 0:
            print(f"  Fast engagement (≤{rb_timing['engagement_hours'].median():.1f}h): Max depth {rb_fast['max_depth'].max()}")
            print(f"  Slow engagement (>{rb_timing['engagement_hours'].median():.1f}h): Max depth {rb_slow['max_depth'].max()}")
            print(f"  Deep conversations (>15 levels): {(rb_fast['max_depth'] > 15).sum()} fast, {(rb_slow['max_depth'] > 15).sum()} slow")
        
        print(f"Poor Branching:")
        if len(pb_fast) > 0 and len(pb_slow) > 0:
            print(f"  Fast engagement (≤{pb_timing['engagement_hours'].median():.1f}h): Max depth {pb_fast['max_depth'].max()}")
            print(f"  Slow engagement (>{pb_timing['engagement_hours'].median():.1f}h): Max depth {pb_slow['max_depth'].max()}")
            print(f"  Deep conversations (>10 levels): {(pb_fast['max_depth'] > 10).sum()} fast, {(pb_slow['max_depth'] > 10).sum()} slow")
        
        # Quality of early vs late conversations
        early_cutoff = 18  # hours
        print(f"\nEARLY vs LATE CONVERSATION QUALITY:")
        print("-" * 35)
        
        rb_early = rb_timing[rb_timing['engagement_hours'] <= early_cutoff]
        rb_late = rb_timing[rb_timing['engagement_hours'] > early_cutoff]
        pb_early = pb_timing[pb_timing['engagement_hours'] <= early_cutoff]
        pb_late = pb_timing[pb_timing['engagement_hours'] > early_cutoff]
        
        print(f"Conversations reaching 80% engagement by {early_cutoff}h:")
        if len(rb_early) > 0:
            deep_early_rb = (rb_early['max_depth'] > 12).sum()
            print(f"  Rich branching: {len(rb_early)} conversations, {deep_early_rb} deep (>12 levels), max depth {rb_early['max_depth'].max()}")
        if len(pb_early) > 0:
            deep_early_pb = (pb_early['max_depth'] > 8).sum()
            print(f"  Poor branching: {len(pb_early)} conversations, {deep_early_pb} deep (>8 levels), max depth {pb_early['max_depth'].max()}")
        
        print(f"Conversations taking >{early_cutoff}h to reach 80% engagement:")
        if len(rb_late) > 0:
            deep_late_rb = (rb_late['max_depth'] > 12).sum()
            print(f"  Rich branching: {len(rb_late)} conversations, {deep_late_rb} deep (>12 levels), max depth {rb_late['max_depth'].max()}")
        if len(pb_late) > 0:
            deep_late_pb = (pb_late['max_depth'] > 8).sum()
            print(f"  Poor branching: {len(pb_late)} conversations, {deep_late_pb} deep (>8 levels), max depth {pb_late['max_depth'].max()}")
    
    # Depth vs Timing correlation analysis
    if len(timing_df_clean) > 5:
        print(f"\nDEPTH vs TIMING CORRELATION:")
        print("-" * 28)
        
        correlation = timing_df_clean['max_depth'].corr(timing_df_clean['engagement_hours'])
        print(f"Correlation between max depth and engagement time: {correlation:.3f}")
        
        # Separate correlations by type
        if len(rb_timing) > 2:
            rb_correlation = rb_timing['max_depth'].corr(rb_timing['engagement_hours'])
            print(f"Rich branching depth-time correlation: {rb_correlation:.3f}")
        
        if len(pb_timing) > 2:
            pb_correlation = pb_timing['max_depth'].corr(pb_timing['engagement_hours'])
            print(f"Poor branching depth-time correlation: {pb_correlation:.3f}")

def main():
    """
    Main function to run the enhanced TST analysis with timing
    """
    print("Starting Enhanced Top Subtree (TST) Analysis with 80% Engagement Timing...")
    
    # Load and analyze conversations
    tst_data, timing_summary = load_and_analyze_conversations()
    
    if tst_data:
        print(f"Successfully analyzed {len(tst_data)} top subtrees")
        
        # Create enhanced visualizations
        create_enhanced_visualizations(tst_data, timing_summary)
        
        # Print enhanced summary
        print_enhanced_summary(tst_data, timing_summary)
        
        # Save data to CSV with timing information
        df = pd.DataFrame(tst_data)
        csv_filename = os.path.join(graphs, 'enhanced_tst_metrics_with_timing.csv')
        df.to_csv(csv_filename, index=False)
        print(f"Saved enhanced TST data: {csv_filename}")
        
        # Save timing summary
        timing_df = pd.DataFrame(timing_summary)
        timing_csv = os.path.join(graphs, 'engagement_timing_summary.csv')
        timing_df.to_csv(timing_csv, index=False)
        print(f"Saved timing summary: {timing_csv}")
        
    else:
        print("No data available for analysis")

if __name__ == "__main__":
    main()