"""
Reddit Discourse Marker Analyzer - Complete Implementation
Analyzes Reddit conversations using IAC v2 discourse markers with subtree analysis
FIXED: Ensures only ONE subtree per post in deepest subtree analysis
"""

import os
import re
import json
import csv
import pandas as pd
from typing import Dict, List, Optional

class IAC2DiscourseMarkerAnalyzer:
    """
    Discourse marker analyzer based on IAC v2 validated research findings
    """
    
    def __init__(self):
        # IAC v2's empirically validated discourse marker rates
        self.disagreement_markers = {
            'really': 0.67,      # 67% of posts starting with "really" signal disagreement
            'no': 0.66,          # 66% disagreement rate
            'actually': 0.60,    # 60% disagreement rate  
            'but': 0.58,         # 58% disagreement rate
            'so': 0.58,          # 58% disagreement rate
            'you mean': 0.57     # 57% disagreement rate
        }
        
        self.agreement_markers = {
            'yes': 0.73,         # 73% of posts starting with "yes" signal agreement
            'i know': 0.64,      # 64% agreement rate
            'i believe': 0.62,   # 62% agreement rate
            'i think': 0.61,     # 61% agreement rate
            'just': 0.57         # 57% agreement rate
        }
        
        self.sarcasm_markers = {
            'you mean': 0.31,    # 31% of posts with "you mean" are sarcastic (highest)
            'oh': 0.29,          # 29% sarcasm rate
            'really': 0.24,      # 24% sarcasm rate
            'so': 0.22,          # 22% sarcasm rate
            'i see': 0.21        # 21% sarcasm rate
        }
        
        # Weak/baseline markers (close to 50% disagreement)
        self.weak_markers = {
            'and': 0.50,      
            'because': 0.51,  
            'oh': 0.51,       
            'i see': 0.52,    
            'you know': 0.54, 
            'well': 0.55      
        }
    
    def analyze_comment(self, comment_text: str) -> Dict:
        """
        Test a single comment for all discourse marker categories
        """
        if not comment_text or comment_text.strip() == '[deleted]':
            return {
                'text': comment_text,
                'valid_comment': False,
                'disagreement_markers': [],
                'agreement_markers': [],
                'sarcasm_markers': [],
                'weak_markers': []
            }
        
        # Clean and normalize text
        text_clean = comment_text.strip().lower()
        text_normalized = re.sub(r'^[^\w\s]+', '', text_clean).strip()
        
        analysis = {
            'text': comment_text,
            'valid_comment': True,
            'disagreement_markers': [],
            'agreement_markers': [],
            'sarcasm_markers': [],
            'weak_markers': []
        }
        
        # Test for disagreement markers (at start of comment)
        for marker, rate in self.disagreement_markers.items():
            if text_normalized.startswith(marker):
                analysis['disagreement_markers'].append({
                    'marker': marker,
                    'disagreement_rate': rate,
                    'position': 'start'
                })
        
        # Test for agreement markers (at start of comment)
        for marker, rate in self.agreement_markers.items():
            if text_normalized.startswith(marker):
                analysis['agreement_markers'].append({
                    'marker': marker,
                    'agreement_rate': rate,
                    'position': 'start'
                })
        
        # Test for sarcasm markers (anywhere in comment)
        for marker, rate in self.sarcasm_markers.items():
            if marker in text_normalized:
                position = 'start' if text_normalized.startswith(marker) else 'within'
                analysis['sarcasm_markers'].append({
                    'marker': marker,
                    'sarcasm_rate': rate,
                    'position': position
                })
        
        # Test for weak markers (at start of comment, only if no strong markers found)
        has_strong_markers = (analysis['disagreement_markers'] or 
                            analysis['agreement_markers'])
        
        if not has_strong_markers:
            for marker, rate in self.weak_markers.items():
                if text_normalized.startswith(marker):
                    analysis['weak_markers'].append({
                        'marker': marker,
                        'disagreement_rate': rate,
                        'position': 'start'
                    })
        
        return analysis


class RedditCommentProcessor:
    """
    Processes Reddit conversation files and analyzes comments for discourse markers
    """
    
    def __init__(self):
        self.discourse_analyzer = IAC2DiscourseMarkerAnalyzer()
    
    def load_conversations_from_folder(self, folder_path):
        """Load all conversation JSON files from the specified folder"""
        conversations = []
        
        if not os.path.exists(folder_path):
            print(f"Error: Folder '{folder_path}' not found.")
            return conversations
        
        json_files = [f for f in os.listdir(folder_path) 
                      if f.endswith('_reddit_comments_with_time.json')]
        
        print(f"Found {len(json_files)} conversation files")
        
        for json_file in json_files:
            file_path = os.path.join(folder_path, json_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                post_id = json_file.replace('_reddit_comments_with_time.json', '')
                
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
    
    def cap_conversation_at_level4(self, data):
        """Cap conversation data at level 4 (depth 4) by removing deeper comments"""
        def filter_comments_by_depth(comments, max_depth=4):
            filtered_comments = []
            
            for comment in comments:
                if comment.get('depth', 0) <= max_depth:
                    filtered_comment = comment.copy()
                    
                    if comment.get('replies') and comment.get('depth', 0) < max_depth:
                        filtered_comment['replies'] = filter_comments_by_depth(comment['replies'], max_depth)
                    else:
                        filtered_comment['replies'] = []
                    
                    filtered_comments.append(filtered_comment)
            
            return filtered_comments
        
        capped_data = data.copy()
        
        if 'comments' in capped_data:
            capped_data['comments'] = filter_comments_by_depth(capped_data['comments'], max_depth=4)
        
        original_title = capped_data.get('post_title', 'Unknown')
        capped_data['post_title'] = f"{original_title} (Level-4 Capped)"
        
        return capped_data
    
    def flatten_reddit_comments(self, comments, parent_id=None):
        """
        Flatten nested Reddit comments into a list suitable for discourse analysis
        """
        flattened = []
        
        for comment in comments:
            # Create post entry for discourse analysis
            post_entry = {
                'comment_id': comment.get('id', ''),
                'text': comment.get('body', ''),
                'author': comment.get('author', 'unknown'),
                'score': comment.get('score', 0),
                'depth': comment.get('depth', 0),
                'timestamp': comment.get('created_utc', ''),
                'parent_id': parent_id,
                'subreddit': comment.get('subreddit', ''),
                'permalink': comment.get('permalink', '')
            }
            
            flattened.append(post_entry)
            
            # Recursively process replies
            if comment.get('replies'):
                child_comments = self.flatten_reddit_comments(
                    comment['replies'], 
                    parent_id=comment.get('id', '')
                )
                flattened.extend(child_comments)
        
        return flattened
    
    def analyze_reddit_conversation(self, conversation_data, cap_at_level4=True):
        """
        Analyze a single Reddit conversation for discourse markers
        """
        # Optionally cap at level 4
        if cap_at_level4:
            conversation_data = self.cap_conversation_at_level4(conversation_data)
        
        # Flatten the comments
        flattened_comments = self.flatten_reddit_comments(conversation_data.get('comments', []))
        
        # Filter out empty or deleted comments
        valid_comments = [c for c in flattened_comments 
                         if c['text'] and c['text'].strip() and c['text'] != '[deleted]']
        
        if not valid_comments:
            return {
                'error': 'No valid comments found in conversation',
                'post_title': conversation_data.get('post_title', 'Unknown'),
                'total_comments': len(flattened_comments)
            }
        
        # Analyze discourse markers
        results = []
        
        for comment in valid_comments:
            analysis = self.discourse_analyzer.analyze_comment(comment['text'])
            
            # Add Reddit-specific metadata
            analysis.update({
                'comment_id': comment['comment_id'],
                'author': comment['author'],
                'score': comment['score'],
                'depth': comment['depth'],
                'timestamp': comment['timestamp'],
                'parent_id': comment['parent_id'],
                'subreddit': comment['subreddit']
            })
            
            results.append(analysis)
        
        # Calculate conversation-level statistics
        summary_stats = self._calculate_summary_stats(results)
        
        return {
            'post_title': conversation_data.get('post_title', 'Unknown'),
            'post_id': conversation_data.get('post_id', 'Unknown'),
            'subreddit': conversation_data.get('subreddit', 'Unknown'),
            'total_comments': len(valid_comments),
            'comment_analyses': results,
            'summary_stats': summary_stats
        }
    
    def _calculate_summary_stats(self, results):
        """Calculate summary statistics for the conversation"""
        if not results:
            return {}
        
        total_comments = len(results)
        
        # Count comments with each marker type
        disagreement_count = sum(1 for c in results if c['disagreement_markers'])
        agreement_count = sum(1 for c in results if c['agreement_markers'])
        sarcasm_count = sum(1 for c in results if c['sarcasm_markers'])
        weak_count = sum(1 for c in results if c['weak_markers'])
        
        # Collect all markers for frequency analysis
        all_disagreement = []
        all_agreement = []
        all_sarcasm = []
        all_weak = []
        
        for comment in results:
            all_disagreement.extend([m['marker'] for m in comment['disagreement_markers']])
            all_agreement.extend([m['marker'] for m in comment['agreement_markers']])
            all_sarcasm.extend([m['marker'] for m in comment['sarcasm_markers']])
            all_weak.extend([m['marker'] for m in comment['weak_markers']])
        
        return {
            'total_comments': total_comments,
            'comments_with_disagreement_markers': disagreement_count,
            'comments_with_agreement_markers': agreement_count,
            'comments_with_sarcasm_markers': sarcasm_count,
            'comments_with_weak_markers': weak_count,
            'disagreement_marker_rate': disagreement_count / total_comments,
            'agreement_marker_rate': agreement_count / total_comments,
            'sarcasm_marker_rate': sarcasm_count / total_comments,
            'weak_marker_rate': weak_count / total_comments,
            'most_common_disagreement_markers': self._count_markers(all_disagreement),
            'most_common_agreement_markers': self._count_markers(all_agreement),
            'most_common_sarcasm_markers': self._count_markers(all_sarcasm),
            'most_common_weak_markers': self._count_markers(all_weak),
            'avg_depth': sum(c['depth'] for c in results) / total_comments,
            'max_depth': max(c['depth'] for c in results),
            'avg_score': sum(c['score'] for c in results) / total_comments
        }
    
    def _count_markers(self, marker_list):
        """Count frequency of markers"""
        counts = {}
        for marker in marker_list:
            counts[marker] = counts.get(marker, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    
    def analyze_all_conversations(self, folder_path, cap_at_level4=True):
        """Analyze all conversations in the folder"""
        conversations = self.load_conversations_from_folder(folder_path)
        
        if not conversations:
            print("No conversations found to analyze")
            return {}
        
        results = {}
        
        for conv in conversations:
            print(f"\nAnalyzing {conv['post_id']} ({conv['conversation_type']})")
            
            analysis = self.analyze_reddit_conversation(conv['data'], cap_at_level4)
            
            # Add conversation metadata
            analysis['conversation_type'] = conv['conversation_type']
            analysis['file_path'] = conv['file_path']
            
            results[conv['post_id']] = analysis
            
            # Print basic stats
            if 'summary_stats' in analysis:
                stats = analysis['summary_stats']
                print(f"  Comments: {stats['total_comments']}")
                print(f"  Disagreement: {stats['disagreement_marker_rate']:.1%}")
                print(f"  Agreement: {stats['agreement_marker_rate']:.1%}")
                print(f"  Sarcasm: {stats['sarcasm_marker_rate']:.1%}")
        
        return results
    
    def analyze_subtrees(self, conversation_data, cap_at_level4=True):
        """
        Analyze individual subtrees (top-level comments and their reply chains)
        """
        # Optionally cap at level 4
        if cap_at_level4:
            conversation_data = self.cap_conversation_at_level4(conversation_data)
        
        subtree_analyses = {}
        
        # Process each top-level comment as a separate subtree
        for i, top_comment in enumerate(conversation_data.get('comments', [])):
            if top_comment.get('depth', 0) == 0:  # Only top-level comments
                subtree_id = f"subtree_{i}_{top_comment.get('id', f'unknown_{i}')}"
                
                # Extract this subtree (top comment + all its nested replies)
                subtree_comments = self.flatten_reddit_comments([top_comment])
                
                # Filter valid comments
                valid_comments = [c for c in subtree_comments 
                                if c['text'] and c['text'].strip() and c['text'] != '[deleted]']
                
                if not valid_comments:
                    continue
                
                # Analyze discourse markers for this subtree
                subtree_results = []
                
                for comment in valid_comments:
                    analysis = self.discourse_analyzer.analyze_comment(comment['text'])
                    
                    # Add comment metadata
                    analysis.update({
                        'comment_id': comment['comment_id'],
                        'author': comment['author'],
                        'score': comment['score'],
                        'depth': comment['depth'],
                        'timestamp': comment['timestamp'],
                        'parent_id': comment['parent_id'],
                        'subreddit': comment['subreddit']
                    })
                    
                    subtree_results.append(analysis)
                
                # Calculate subtree statistics
                subtree_stats = self._calculate_summary_stats(subtree_results)
                
                subtree_analyses[subtree_id] = {
                    'top_comment_id': top_comment.get('id', ''),
                    'top_comment_author': top_comment.get('author', 'unknown'),
                    'top_comment_text': top_comment.get('body', ''),
                    'top_comment_score': top_comment.get('score', 0),
                    'max_depth': max(c['depth'] for c in subtree_results) if subtree_results else 0,
                    'total_comments': len(subtree_results),
                    'comment_analyses': subtree_results,
                    'subtree_stats': subtree_stats
                }
        
        return subtree_analyses
    
    def analyze_all_conversations_with_subtrees(self, folder_path, cap_at_level4=True):
        """
        Analyze all conversations and their subtrees
        """
        conversations = self.load_conversations_from_folder(folder_path)
        
        if not conversations:
            print("No conversations found to analyze")
            return {}, {}
        
        conversation_results = {}
        all_subtree_results = {}
        
        for conv in conversations:
            print(f"\nAnalyzing {conv['post_id']} ({conv['conversation_type']})")
            
            # Regular conversation analysis
            conversation_analysis = self.analyze_reddit_conversation(conv['data'], cap_at_level4)
            conversation_analysis['conversation_type'] = conv['conversation_type']
            conversation_analysis['file_path'] = conv['file_path']
            conversation_results[conv['post_id']] = conversation_analysis
            
            # Subtree analysis
            subtree_analysis = self.analyze_subtrees(conv['data'], cap_at_level4)
            
            # Add conversation metadata to each subtree result
            conversation_metadata = {
                'post_id': conv['post_id'],
                'conversation_type': conv['conversation_type']
            }
            
            all_subtree_results[conv['post_id']] = {
                'metadata': conversation_metadata,
                'subtrees': subtree_analysis
            }
            
            # Print basic stats
            if 'summary_stats' in conversation_analysis:
                stats = conversation_analysis['summary_stats']
                print(f"  Total comments: {stats['total_comments']}")
                print(f"  Subtrees: {len(subtree_analysis)}")
                print(f"  Disagreement rate: {stats['disagreement_marker_rate']:.1%}")
                print(f"  Agreement rate: {stats['agreement_marker_rate']:.1%}")
        
        return conversation_results, all_subtree_results
    
    def save_results(self, results, output_file):
        """Save analysis results to JSON file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_file}")
    
    def save_csv_results(self, results, output_file):
        """Save analysis results to CSV file with highest marker detection"""
        csv_data = []
        
        for post_id, analysis in results.items():
            if 'comment_analyses' in analysis:
                conversation_type = analysis.get('conversation_type', 'unknown')
                
                for comment_analysis in analysis['comment_analyses']:
                    # Extract marker information with rates
                    disagreement_markers = comment_analysis.get('disagreement_markers', [])
                    agreement_markers = comment_analysis.get('agreement_markers', [])
                    sarcasm_markers = comment_analysis.get('sarcasm_markers', [])
                    weak_markers = comment_analysis.get('weak_markers', [])
                    
                    # Find the highest marker across all categories
                    all_markers = []
                    
                    # Add disagreement markers with their rates
                    for m in disagreement_markers:
                        all_markers.append({
                            'marker': m['marker'],
                            'rate': m.get('disagreement_rate', 0),
                            'category': 'disagreement'
                        })
                    
                    # Add agreement markers with their rates  
                    for m in agreement_markers:
                        all_markers.append({
                            'marker': m['marker'],
                            'rate': m.get('agreement_rate', 0),
                            'category': 'agreement'
                        })
                    
                    # Add sarcasm markers with their rates
                    for m in sarcasm_markers:
                        all_markers.append({
                            'marker': m['marker'],
                            'rate': m.get('sarcasm_rate', 0),
                            'category': 'sarcasm'
                        })
                    
                    # Add weak markers with their rates
                    for m in weak_markers:
                        all_markers.append({
                            'marker': m['marker'],
                            'rate': m.get('disagreement_rate', 0),
                            'category': 'weak'
                        })
                    
                    # Find highest marker
                    highest_marker = ''
                    highest_rate = 0
                    highest_category = ''
                    
                    if all_markers:
                        # Sort by rate and get the highest
                        highest = max(all_markers, key=lambda x: x['rate'])
                        highest_marker = highest['marker']
                        highest_rate = highest['rate']
                        highest_category = highest['category']
                    
                    # Create row for CSV
                    row = {
                        'post_id': post_id,
                        'conversation_type': conversation_type,
                        'comment_id': comment_analysis.get('comment_id', ''),
                        'author': comment_analysis.get('author', ''),
                        'depth': comment_analysis.get('depth', 0),
                        'score': comment_analysis.get('score', 0),
                        'text': comment_analysis.get('text', '').replace('\n', ' ').replace('\r', ' ')[:100] + '...' if len(comment_analysis.get('text', '')) > 100 else comment_analysis.get('text', ''),
                        'disagreement_markers': '; '.join([m['marker'] for m in disagreement_markers]),
                        'agreement_markers': '; '.join([m['marker'] for m in agreement_markers]),
                        'sarcasm_markers': '; '.join([m['marker'] for m in sarcasm_markers]),
                        'weak_markers': '; '.join([m['marker'] for m in weak_markers]),
                        'highest_marker': highest_marker,
                        'highest_rate': f"{highest_rate:.3f}" if highest_rate > 0 else '',
                        'highest_category': highest_category,
                        'total_markers': len(all_markers)
                    }
                    
                    csv_data.append(row)
        
        # Write to CSV
        if csv_data:
            fieldnames = ['post_id', 'conversation_type', 'comment_id', 'author', 'depth', 'score', 
                         'text', 'disagreement_markers', 'agreement_markers', 'sarcasm_markers', 
                         'weak_markers', 'highest_marker', 'highest_rate', 'highest_category', 'total_markers']
            
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
            
            print(f"CSV results saved to: {output_file}")
            print(f"Total rows: {len(csv_data)}")
        else:
            print("No data to save to CSV")
    
    def generate_summary_csv(self, results, output_file):
        """Generate a conversation-level summary CSV"""
        summary_data = []
        
        for post_id, analysis in results.items():
            if 'summary_stats' in analysis:
                stats = analysis['summary_stats']
                
                row = {
                    'post_id': post_id,
                    'conversation_type': analysis.get('conversation_type', 'unknown'),
                    'total_comments': stats.get('total_comments', 0),
                    'comments_with_disagreement_markers': stats.get('comments_with_disagreement_markers', 0),
                    'comments_with_agreement_markers': stats.get('comments_with_agreement_markers', 0),
                    'comments_with_sarcasm_markers': stats.get('comments_with_sarcasm_markers', 0),
                    'comments_with_weak_markers': stats.get('comments_with_weak_markers', 0),
                    'disagreement_marker_rate': f"{stats.get('disagreement_marker_rate', 0):.3f}",
                    'agreement_marker_rate': f"{stats.get('agreement_marker_rate', 0):.3f}",
                    'sarcasm_marker_rate': f"{stats.get('sarcasm_marker_rate', 0):.3f}",
                    'weak_marker_rate': f"{stats.get('weak_marker_rate', 0):.3f}",
                    'avg_depth': f"{stats.get('avg_depth', 0):.2f}",
                    'avg_score': f"{stats.get('avg_score', 0):.2f}"
                }
                
                summary_data.append(row)
        
        # Write summary CSV
        if summary_data:
            fieldnames = ['post_id', 'conversation_type', 'total_comments', 
                         'comments_with_disagreement_markers', 'comments_with_agreement_markers',
                         'comments_with_sarcasm_markers', 'comments_with_weak_markers',
                         'disagreement_marker_rate', 'agreement_marker_rate', 
                         'sarcasm_marker_rate', 'weak_marker_rate', 'avg_depth', 'avg_score']
            
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary_data)
            
            print(f"Summary CSV saved to: {output_file}")
            print(f"Total conversations: {len(summary_data)}")
    
    def save_subtree_csv(self, subtree_results, conversation_metadata, output_file):
        """Save subtree analysis results to CSV file"""
        csv_data = []
        
        for subtree_id, subtree_analysis in subtree_results.items():
            for comment_analysis in subtree_analysis['comment_analyses']:
                # Extract marker information with rates
                disagreement_markers = comment_analysis.get('disagreement_markers', [])
                agreement_markers = comment_analysis.get('agreement_markers', [])
                sarcasm_markers = comment_analysis.get('sarcasm_markers', [])
                weak_markers = comment_analysis.get('weak_markers', [])
                
                # Find the highest marker across all categories
                all_markers = []
                
                # Add disagreement markers with their rates
                for m in disagreement_markers:
                    all_markers.append({
                        'marker': m['marker'],
                        'rate': m.get('disagreement_rate', 0),
                        'category': 'disagreement'
                    })
                
                # Add agreement markers with their rates  
                for m in agreement_markers:
                    all_markers.append({
                        'marker': m['marker'],
                        'rate': m.get('agreement_rate', 0),
                        'category': 'agreement'
                    })
                
                # Add sarcasm markers with their rates
                for m in sarcasm_markers:
                    all_markers.append({
                        'marker': m['marker'],
                        'rate': m.get('sarcasm_rate', 0),
                        'category': 'sarcasm'
                    })
                
                # Add weak markers with their rates
                for m in weak_markers:
                    all_markers.append({
                        'marker': m['marker'],
                        'rate': m.get('disagreement_rate', 0),
                        'category': 'weak'
                    })
                
                # Find highest marker
                highest_marker = ''
                highest_rate = 0
                highest_category = ''
                
                if all_markers:
                    highest = max(all_markers, key=lambda x: x['rate'])
                    highest_marker = highest['marker']
                    highest_rate = highest['rate']
                    highest_category = highest['category']
                
                # Create row for CSV
                row = {
                    'post_id': conversation_metadata.get('post_id', 'unknown'),
                    'conversation_type': conversation_metadata.get('conversation_type', 'unknown'),
                    'subtree_id': subtree_id,
                    'top_comment_id': subtree_analysis['top_comment_id'],
                    'top_comment_author': subtree_analysis['top_comment_author'],
                    'subtree_max_depth': subtree_analysis['max_depth'],
                    'subtree_total_comments': subtree_analysis['total_comments'],
                    'comment_id': comment_analysis.get('comment_id', ''),
                    'author': comment_analysis.get('author', ''),
                    'depth': comment_analysis.get('depth', 0),
                    'score': comment_analysis.get('score', 0),
                    'text': comment_analysis.get('text', '').replace('\n', ' ').replace('\r', ' ')[:100] + '...' if len(comment_analysis.get('text', '')) > 100 else comment_analysis.get('text', ''),
                    'disagreement_markers': '; '.join([m['marker'] for m in disagreement_markers]),
                    'agreement_markers': '; '.join([m['marker'] for m in agreement_markers]),
                    'sarcasm_markers': '; '.join([m['marker'] for m in sarcasm_markers]),
                    'weak_markers': '; '.join([m['marker'] for m in weak_markers]),
                    'highest_marker': highest_marker,
                    'highest_rate': f"{highest_rate:.3f}" if highest_rate > 0 else '',
                    'highest_category': highest_category,
                    'total_markers': len(all_markers)
                }
                
                csv_data.append(row)
        
        # Write to CSV
        if csv_data:
            fieldnames = ['post_id', 'conversation_type', 'subtree_id', 'top_comment_id', 'top_comment_author', 
                         'subtree_max_depth', 'subtree_total_comments', 'comment_id', 'author', 'depth', 'score', 
                         'text', 'disagreement_markers', 'agreement_markers', 'sarcasm_markers', 
                         'weak_markers', 'highest_marker', 'highest_rate', 'highest_category', 'total_markers']
            
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
            
            print(f"Subtree CSV results saved to: {output_file}")
            print(f"Total rows: {len(csv_data)}")
        else:
            print("No subtree data to save to CSV")
    
    def generate_subtree_summary_csv(self, subtree_results, conversation_metadata, output_file):
        """Generate a subtree-level summary CSV"""
        summary_data = []
        
        for subtree_id, subtree_analysis in subtree_results.items():
            if 'subtree_stats' in subtree_analysis:
                stats = subtree_analysis['subtree_stats']
                
                row = {
                    'post_id': conversation_metadata.get('post_id', 'unknown'),
                    'conversation_type': conversation_metadata.get('conversation_type', 'unknown'),
                    'subtree_id': subtree_id,
                    'top_comment_id': subtree_analysis['top_comment_id'],
                    'top_comment_author': subtree_analysis['top_comment_author'],
                    'top_comment_score': subtree_analysis['top_comment_score'],
                    'max_depth': subtree_analysis['max_depth'],
                    'total_comments': stats.get('total_comments', 0),
                    'comments_with_disagreement_markers': stats.get('comments_with_disagreement_markers', 0),
                    'comments_with_agreement_markers': stats.get('comments_with_agreement_markers', 0),
                    'comments_with_sarcasm_markers': stats.get('comments_with_sarcasm_markers', 0),
                    'comments_with_weak_markers': stats.get('comments_with_weak_markers', 0),
                    'disagreement_marker_rate': f"{stats.get('disagreement_marker_rate', 0):.3f}",
                    'agreement_marker_rate': f"{stats.get('agreement_marker_rate', 0):.3f}",
                    'sarcasm_marker_rate': f"{stats.get('sarcasm_marker_rate', 0):.3f}",
                    'weak_marker_rate': f"{stats.get('weak_marker_rate', 0):.3f}",
                    'avg_depth': f"{stats.get('avg_depth', 0):.2f}",
                    'avg_score': f"{stats.get('avg_score', 0):.2f}"
                }
                
                summary_data.append(row)
        
        # Write summary CSV
        if summary_data:
            fieldnames = ['post_id', 'conversation_type', 'subtree_id', 'top_comment_id', 'top_comment_author',
                         'top_comment_score', 'max_depth', 'total_comments', 
                         'comments_with_disagreement_markers', 'comments_with_agreement_markers',
                         'comments_with_sarcasm_markers', 'comments_with_weak_markers',
                         'disagreement_marker_rate', 'agreement_marker_rate', 
                         'sarcasm_marker_rate', 'weak_marker_rate', 'avg_depth', 'avg_score']
            
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary_data)
            
            print(f"Subtree summary CSV saved to: {output_file}")
            print(f"Total subtrees: {len(summary_data)}")

    def generate_deepest_subtree_summary_csv(self, subtree_results, output_file):
        """
        Generate a subtree summary CSV with only the DEEPEST subtree per post
        to match the structure of original_posts_discourse_markers_summary.csv
        FIXED: Ensures only ONE subtree per post_id
        """
        # Collect all subtree data with metadata
        all_subtree_data = []
        
        for post_id, subtree_data in subtree_results.items():
            conversation_metadata = subtree_data['metadata']
            subtrees = subtree_data['subtrees']
            
            for subtree_id, subtree_analysis in subtrees.items():
                if 'subtree_stats' in subtree_analysis:
                    stats = subtree_analysis['subtree_stats']
                    
                    subtree_info = {
                        'post_id': conversation_metadata.get('post_id', 'unknown'),
                        'conversation_type': conversation_metadata.get('conversation_type', 'unknown'),
                        'subtree_id': subtree_id,
                        'max_depth': subtree_analysis['max_depth'],
                        'total_comments': stats.get('total_comments', 0),
                        'comments_with_disagreement_markers': stats.get('comments_with_disagreement_markers', 0),
                        'comments_with_agreement_markers': stats.get('comments_with_agreement_markers', 0),
                        'comments_with_sarcasm_markers': stats.get('comments_with_sarcasm_markers', 0),
                        'comments_with_weak_markers': stats.get('comments_with_weak_markers', 0),
                        'disagreement_marker_rate': stats.get('disagreement_marker_rate', 0),
                        'agreement_marker_rate': stats.get('agreement_marker_rate', 0),
                        'sarcasm_marker_rate': stats.get('sarcasm_marker_rate', 0),
                        'weak_marker_rate': stats.get('weak_marker_rate', 0),
                        'avg_depth': stats.get('avg_depth', 0),
                        'avg_score': stats.get('avg_score', 0)
                    }
                    
                    all_subtree_data.append(subtree_info)
        
        # Convert to DataFrame
        all_subtrees_df = pd.DataFrame(all_subtree_data)
        
        if len(all_subtrees_df) == 0:
            print("No subtree data to process")
            return
        
        # DEBUG: Print subtree counts per post
        print(f"DEBUG: Subtree counts per post:")
        subtree_counts = all_subtrees_df.groupby('post_id').size()
        for post_id, count in subtree_counts.items():
            print(f"  {post_id}: {count} subtrees")
        
        # Find the deepest subtree for each post_id - WITH TIE-BREAKING
        print(f"\nFinding deepest subtree for each post...")
        deepest_subtrees = []
        
        for post_id in all_subtrees_df['post_id'].unique():
            post_subtrees = all_subtrees_df[all_subtrees_df['post_id'] == post_id].copy()
            
            # Find maximum depth for this post
            max_depth = post_subtrees['max_depth'].max()
            
            # Get all subtrees with maximum depth
            max_depth_subtrees = post_subtrees[post_subtrees['max_depth'] == max_depth]
            
            print(f"  Post {post_id}: {len(max_depth_subtrees)} subtrees with max depth {max_depth}")
            
            # If multiple subtrees have same max depth, pick the one with most comments
            if len(max_depth_subtrees) > 1:
                print(f"    TIE DETECTED: Multiple subtrees with depth {max_depth}")
                # Tie-breaker 1: Most comments
                max_comments = max_depth_subtrees['total_comments'].max()
                tie_breakers = max_depth_subtrees[max_depth_subtrees['total_comments'] == max_comments]
                
                if len(tie_breakers) > 1:
                    print(f"    TIE STILL EXISTS: Multiple subtrees with {max_comments} comments")
                    # Tie-breaker 2: Highest disagreement marker rate
                    max_disagreement = tie_breakers['disagreement_marker_rate'].max()
                    final_candidates = tie_breakers[tie_breakers['disagreement_marker_rate'] == max_disagreement]
                    
                    if len(final_candidates) > 1:
                        print(f"    FINAL TIE-BREAKER: Taking first alphabetically by subtree_id")
                        # Final tie-breaker: First alphabetically by subtree_id
                        chosen_subtree = final_candidates.loc[final_candidates['subtree_id'].idxmin()]
                    else:
                        chosen_subtree = final_candidates.iloc[0]
                else:
                    chosen_subtree = tie_breakers.iloc[0]
            else:
                chosen_subtree = max_depth_subtrees.iloc[0]
            
            print(f"    SELECTED: {chosen_subtree['subtree_id']} "
                  f"(depth={chosen_subtree['max_depth']}, "
                  f"comments={chosen_subtree['total_comments']}, "
                  f"disagreement_rate={chosen_subtree['disagreement_marker_rate']:.3f})")
            
            deepest_subtrees.append(chosen_subtree)
        
        # Convert to DataFrame and verify uniqueness
        deepest_df = pd.DataFrame(deepest_subtrees)
        
        # VERIFICATION: Check for duplicates
        duplicate_posts = deepest_df['post_id'].duplicated()
        if duplicate_posts.any():
            print(f"ERROR: Found duplicate post_ids after selection!")
            duplicates = deepest_df[duplicate_posts]['post_id'].tolist()
            print(f"Duplicate post_ids: {duplicates}")
            return None
        else:
            print(f"✅ VERIFICATION PASSED: {len(deepest_df)} unique posts selected")
        
        # Select columns that match original posts summary format
        summary_columns = [
            'post_id',
            'conversation_type',
            'total_comments',
            'comments_with_disagreement_markers',
            'comments_with_agreement_markers',
            'comments_with_sarcasm_markers',
            'comments_with_weak_markers',
            'disagreement_marker_rate',
            'agreement_marker_rate',
            'sarcasm_marker_rate',
            'weak_marker_rate',
            'avg_depth',
            'avg_score'
        ]
        
        # Keep only the columns we want (in case some are missing)
        available_columns = [col for col in summary_columns if col in deepest_df.columns]
        final_summary = deepest_df[available_columns].copy()
        
        # Round the rates to 3 decimal places for consistency
        rate_columns = ['disagreement_marker_rate', 'agreement_marker_rate', 
                       'sarcasm_marker_rate', 'weak_marker_rate']
        for col in rate_columns:
            if col in final_summary.columns:
                final_summary[col] = final_summary[col].round(3)
        
        # Round other float columns
        float_columns = ['avg_depth', 'avg_score']
        for col in float_columns:
            if col in final_summary.columns:
                final_summary[col] = final_summary[col].round(2)
        
        # Write summary CSV
        final_summary.to_csv(output_file, index=False)
        
        print(f"\n✅ Deepest subtree summary saved: {output_file}")
        print(f"📊 Total posts: {len(final_summary)}")
        print(f"📋 Structure matches original_posts_discourse_markers_summary.csv")
        
        # Show distribution by conversation type
        print(f"\nConversation type distribution:")
        type_counts = final_summary['conversation_type'].value_counts()
        for conv_type, count in type_counts.items():
            print(f"  {conv_type}: {count} posts")
        
        # FINAL VERIFICATION: Show post_id uniqueness
        print(f"\n🔍 FINAL VERIFICATION:")
        print(f"   Unique post_ids in output: {final_summary['post_id'].nunique()}")
        print(f"   Total rows in output: {len(final_summary)}")
        print(f"   Match: {'✅ YES' if final_summary['post_id'].nunique() == len(final_summary) else '❌ NO'}")
        
        return final_summary


# Usage and Main Execution
if __name__ == "__main__":
    # Initialize processor
    processor = RedditCommentProcessor()
    
    # Create discourse_subtree folder if it doesn't exist
    subtree_folder = 'discourse_subtree'
    if not os.path.exists(subtree_folder):
        os.makedirs(subtree_folder)
        print(f"Created folder: {subtree_folder}")
    
    # Analyze all conversations AND subtrees
    conversation_results, subtree_results = processor.analyze_all_conversations_with_subtrees('json_data', cap_at_level4=True)
    
    # Save conversation-level results (original posts analysis)
    processor.save_csv_results(conversation_results, 'original_posts_discourse_markers_detailed.csv')
    processor.generate_summary_csv(conversation_results, 'original_posts_discourse_markers_summary.csv')
    
    # Generate combined subtree files only (no individual files)
    all_subtree_csv_data = []
    all_subtree_summary_data = []
    
    for post_id, subtree_data in subtree_results.items():
        # Collect detailed data for all subtrees
        for subtree_id, subtree_analysis in subtree_data['subtrees'].items():
            for comment_analysis in subtree_analysis['comment_analyses']:
                # Extract marker information with rates
                disagreement_markers = comment_analysis.get('disagreement_markers', [])
                agreement_markers = comment_analysis.get('agreement_markers', [])
                sarcasm_markers = comment_analysis.get('sarcasm_markers', [])
                weak_markers = comment_analysis.get('weak_markers', [])
                
                # Find the highest marker across all categories
                all_markers = []
                
                # Add disagreement markers with their rates
                for m in disagreement_markers:
                    all_markers.append({
                        'marker': m['marker'],
                        'rate': m.get('disagreement_rate', 0),
                        'category': 'disagreement'
                    })
                
                # Add agreement markers with their rates  
                for m in agreement_markers:
                    all_markers.append({
                        'marker': m['marker'],
                        'rate': m.get('agreement_rate', 0),
                        'category': 'agreement'
                    })
                
                # Add sarcasm markers with their rates
                for m in sarcasm_markers:
                    all_markers.append({
                        'marker': m['marker'],
                        'rate': m.get('sarcasm_rate', 0),
                        'category': 'sarcasm'
                    })
                
                # Add weak markers with their rates
                for m in weak_markers:
                    all_markers.append({
                        'marker': m['marker'],
                        'rate': m.get('disagreement_rate', 0),
                        'category': 'weak'
                    })
                
                # Find highest marker
                highest_marker = ''
                highest_rate = 0
                highest_category = ''
                
                if all_markers:
                    highest = max(all_markers, key=lambda x: x['rate'])
                    highest_marker = highest['marker']
                    highest_rate = highest['rate']
                    highest_category = highest['category']
                
                # Create row for detailed CSV
                row = {
                    'post_id': subtree_data['metadata'].get('post_id', 'unknown'),
                    'conversation_type': subtree_data['metadata'].get('conversation_type', 'unknown'),
                    'subtree_id': subtree_id,
                    'top_comment_id': subtree_analysis['top_comment_id'],
                    'top_comment_author': subtree_analysis['top_comment_author'],
                    'subtree_max_depth': subtree_analysis['max_depth'],
                    'subtree_total_comments': subtree_analysis['total_comments'],
                    'comment_id': comment_analysis.get('comment_id', ''),
                    'author': comment_analysis.get('author', ''),
                    'depth': comment_analysis.get('depth', 0),
                    'score': comment_analysis.get('score', 0),
                    'text': comment_analysis.get('text', '').replace('\n', ' ').replace('\r', ' ')[:100] + '...' if len(comment_analysis.get('text', '')) > 100 else comment_analysis.get('text', ''),
                    'disagreement_markers': '; '.join([m['marker'] for m in disagreement_markers]),
                    'agreement_markers': '; '.join([m['marker'] for m in agreement_markers]),
                    'sarcasm_markers': '; '.join([m['marker'] for m in sarcasm_markers]),
                    'weak_markers': '; '.join([m['marker'] for m in weak_markers]),
                    'highest_marker': highest_marker,
                    'highest_rate': f"{highest_rate:.3f}" if highest_rate > 0 else '',
                    'highest_category': highest_category,
                    'total_markers': len(all_markers)
                }
                
                all_subtree_csv_data.append(row)
        
        # Collect summary data for all subtrees
        for subtree_id, subtree_analysis in subtree_data['subtrees'].items():
            if 'subtree_stats' in subtree_analysis:
                stats = subtree_analysis['subtree_stats']
                
                summary_row = {
                    'post_id': subtree_data['metadata'].get('post_id', 'unknown'),
                    'conversation_type': subtree_data['metadata'].get('conversation_type', 'unknown'),
                    'subtree_id': subtree_id,
                    'top_comment_id': subtree_analysis['top_comment_id'],
                    'top_comment_author': subtree_analysis['top_comment_author'],
                    'top_comment_score': subtree_analysis['top_comment_score'],
                    'max_depth': subtree_analysis['max_depth'],
                    'total_comments': stats.get('total_comments', 0),
                    'comments_with_disagreement_markers': stats.get('comments_with_disagreement_markers', 0),
                    'comments_with_agreement_markers': stats.get('comments_with_agreement_markers', 0),
                    'comments_with_sarcasm_markers': stats.get('comments_with_sarcasm_markers', 0),
                    'comments_with_weak_markers': stats.get('comments_with_weak_markers', 0),
                    'disagreement_marker_rate': f"{stats.get('disagreement_marker_rate', 0):.3f}",
                    'agreement_marker_rate': f"{stats.get('agreement_marker_rate', 0):.3f}",
                    'sarcasm_marker_rate': f"{stats.get('sarcasm_marker_rate', 0):.3f}",
                    'weak_marker_rate': f"{stats.get('weak_marker_rate', 0):.3f}",
                    'avg_depth': f"{stats.get('avg_depth', 0):.2f}",
                    'avg_score': f"{stats.get('avg_score', 0):.2f}"
                }
                
                all_subtree_summary_data.append(summary_row)
    
    # Save combined subtree files only
    combined_detailed_file = os.path.join(subtree_folder, 'discourse_subtrees_all_detailed.csv')
    if all_subtree_csv_data:
        detailed_fieldnames = ['post_id', 'conversation_type', 'subtree_id', 'top_comment_id', 'top_comment_author', 
                             'subtree_max_depth', 'subtree_total_comments', 'comment_id', 'author', 'depth', 'score', 
                             'text', 'disagreement_markers', 'agreement_markers', 'sarcasm_markers', 
                             'weak_markers', 'highest_marker', 'highest_rate', 'highest_category', 'total_markers']
        
        with open(combined_detailed_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=detailed_fieldnames)
            writer.writeheader()
            writer.writerows(all_subtree_csv_data)
        print(f"Combined subtree detailed CSV saved: {combined_detailed_file}")
    
    combined_summary_file = os.path.join(subtree_folder, 'discourse_subtrees_all_summary.csv')
    if all_subtree_summary_data:
        summary_fieldnames = ['post_id', 'conversation_type', 'subtree_id', 'top_comment_id', 'top_comment_author',
                             'top_comment_score', 'max_depth', 'total_comments', 
                             'comments_with_disagreement_markers', 'comments_with_agreement_markers',
                             'comments_with_sarcasm_markers', 'comments_with_weak_markers',
                             'disagreement_marker_rate', 'agreement_marker_rate', 
                             'sarcasm_marker_rate', 'weak_marker_rate', 'avg_depth', 'avg_score']
        
        with open(combined_summary_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=summary_fieldnames)
            writer.writeheader()
            writer.writerows(all_subtree_summary_data)
        print(f"Combined subtree summary CSV saved: {combined_summary_file}")
    
    # ✨ NEW: Generate deepest subtree summary (matching original posts structure)
    print(f"\n{'='*60}")
    print("GENERATING DEEPEST SUBTREE SUMMARY (Fair Comparison)")
    print(f"{'='*60}")
    
    deepest_summary = processor.generate_deepest_subtree_summary_csv(
        subtree_results, 
        'deepest_subtrees_summary.csv'
    )
    
    # Print comparison info
    print(f"\n{'='*60}")
    print("FAIR COMPARISON READY!")
    print(f"{'='*60}")
    
    print(f"📊 Original Posts Summary: original_posts_discourse_markers_summary.csv")
    print(f"   Structure: One row per post (conversation-level analysis)")
    
    print(f"📊 Deepest Subtrees Summary: deepest_subtrees_summary.csv") 
    print(f"   Structure: One row per post (deepest subtree analysis)")
    
    print(f"\n🎯 FAIR COMPARISON: Both files now have:")
    print(f"   - Same number of rows (one per post)")
    print(f"   - Same column structure")
    print(f"   - Same level-4 depth capping")
    print(f"   - Ready for statistical comparison!")
    
    # Show sample comparison
    try:
        original_summary = pd.read_csv('original_posts_discourse_markers_summary.csv')
        print(f"\n📈 Sample Comparison:")
        print(f"   Original Posts: {len(original_summary)} posts")
        print(f"   Deepest Subtrees: {len(deepest_summary)} posts")
        
        # Show conversation type distribution
        print(f"\n   Original Posts by type:")
        for conv_type, count in original_summary['conversation_type'].value_counts().items():
            print(f"     {conv_type}: {count}")
        
        print(f"\n   Deepest Subtrees by type:")
        for conv_type, count in deepest_summary['conversation_type'].value_counts().items():
            print(f"     {conv_type}: {count}")
            
    except Exception as e:
        print(f"Could not load comparison data: {e}")
    
    # Print overall summary
    print(f"\n{'='*60}")
    print("OVERALL ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    total_conversations = len(conversation_results)
    total_comments = sum(r.get('total_comments', 0) for r in conversation_results.values())
    total_subtrees = sum(len(s['subtrees']) for s in subtree_results.values())
    
    print(f"Total conversations analyzed: {total_conversations}")
    print(f"Total comments analyzed: {total_comments}")
    print(f"Total subtrees identified: {total_subtrees}")
    
    print(f"\nFiles generated:")
    print(f"\n📊 Original Posts Analysis (main folder):")
    print(f"  1. original_posts_discourse_markers_detailed.csv")
    print(f"  2. original_posts_discourse_markers_summary.csv")
    
    print(f"\n🎯 Fair Comparison File (main folder):")
    print(f"  3. deepest_subtrees_summary.csv - DEEPEST SUBTREE per post (Fair comparison)")
    
    print(f"\n🌳 Complete Subtree Analysis ({subtree_folder}/ folder):")
    print(f"  4. discourse_subtrees_all_detailed.csv - All subtree comments")
    print(f"  5. discourse_subtrees_all_summary.csv - All subtree summaries")
    
    # Show sample analysis
    print(f"\n{'='*60}")
    print("SAMPLE ANALYSIS OUTPUT")
    print(f"{'='*60}")
    
    sample_count = 0
    for post_id, analysis in conversation_results.items():
        if 'comment_analyses' in analysis and sample_count < 2:
            print(f"\nConversation: {post_id} ({analysis.get('conversation_type', 'unknown')})")
            
            if 'summary_stats' in analysis:
                stats = analysis['summary_stats']
                print(f"  Total comments: {stats['total_comments']}")
                print(f"  Disagreement rate: {stats['disagreement_marker_rate']:.1%}")
                print(f"  Agreement rate: {stats['agreement_marker_rate']:.1%}")
                print(f"  Sarcasm rate: {stats['sarcasm_marker_rate']:.1%}")
                print(f"  Weak markers rate: {stats['weak_marker_rate']:.1%}")
            
            # Show first few comments with markers
            for comment in analysis['comment_analyses'][:3]:
                if any([comment['disagreement_markers'], comment['agreement_markers'], 
                       comment['sarcasm_markers'], comment['weak_markers']]):
                    
                    # Find highest marker
                    all_markers = []
                    for m in comment['disagreement_markers']:
                        all_markers.append({'marker': m['marker'], 'rate': m.get('disagreement_rate', 0), 'category': 'disagreement'})
                    for m in comment['agreement_markers']:
                        all_markers.append({'marker': m['marker'], 'rate': m.get('agreement_rate', 0), 'category': 'agreement'})
                    for m in comment['sarcasm_markers']:
                        all_markers.append({'marker': m['marker'], 'rate': m.get('sarcasm_rate', 0), 'category': 'sarcasm'})
                    for m in comment['weak_markers']:
                        all_markers.append({'marker': m['marker'], 'rate': m.get('disagreement_rate', 0), 'category': 'weak'})
                    
                    highest = max(all_markers, key=lambda x: x['rate']) if all_markers else None
                    
                    print(f"    Comment (depth {comment['depth']}): {comment['text'][:60]}...")
                    if highest:
                        print(f"    → Highest marker: '{highest['marker']}' ({highest['rate']:.3f}) [{highest['category']}]")
                    print()
            
            sample_count += 1
    
    # Show subtree sample
    if subtree_results:
        print(f"\n{'='*60}")
        print("SUBTREE ANALYSIS SAMPLE")
        print(f"{'='*60}")
        
        sample_post = list(subtree_results.keys())[0]
        sample_subtrees = subtree_results[sample_post]['subtrees']
        
        print(f"\nPost: {sample_post}")
        print(f"Number of subtrees: {len(sample_subtrees)}")
        
        # Show first subtree
        for subtree_id, subtree in list(sample_subtrees.items())[:1]:
            print(f"\nSubtree: {subtree_id}")
            print(f"  Top comment: {subtree['top_comment_text'][:80]}...")
            print(f"  Author: {subtree['top_comment_author']}")
            print(f"  Max depth: {subtree['max_depth']}")
            print(f"  Total comments: {subtree['total_comments']}")
            
            if 'subtree_stats' in subtree:
                stats = subtree['subtree_stats']
                print(f"  Disagreement rate: {stats['disagreement_marker_rate']:.1%}")
                print(f"  Agreement rate: {stats['agreement_marker_rate']:.1%}")
                print(f"  Sarcasm rate: {stats['sarcasm_marker_rate']:.1%}")
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE!")
    print("✅ Fair comparison ready: original_posts_discourse_markers_summary.csv vs deepest_subtrees_summary.csv")
    print("Ready for statistical analysis and manuscript preparation!")
    print(f"{'='*60}")