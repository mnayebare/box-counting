import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# For VADER sentiment analysis
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("ERROR: vaderSentiment not available. Please install with: pip install vaderSentiment")
    print("This analysis requires VADER for optimal Reddit comment sentiment analysis.")
    exit(1)

class TSTSentimentAnalyzer:
    """
    TST sentiment analyzer - finds the deepest subtree and analyzes sentiment within it
    Uses level 4 cap and max-depth based selection
    Preserves individual sentiment values
    MODIFIED: Uses max depth instead of scaling factor formula for TST selection
    """
    
    def __init__(self, max_level=4):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.max_level = max_level
        self.results = []
        
    def analyze_sentiment(self, text):
        """Analyze sentiment using VADER"""
        if not text or text.strip() == "":
            return {
                'compound': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'sentiment_label': 'neutral'
            }
        
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            compound = scores['compound']
            
            # Classify sentiment
            if compound >= 0.05:
                sentiment_label = 'positive'
            elif compound <= -0.05:
                sentiment_label = 'negative'
            else:
                sentiment_label = 'neutral'
            
            return {
                'compound': compound,
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu'],
                'sentiment_label': sentiment_label
            }
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return {
                'compound': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'sentiment_label': 'neutral'
            }

    def calculate_polarization_scores(self, compound_scores, positive_scores, negative_scores):
        """Calculate sentiment strength and polarization metrics using medians and means"""
        if not compound_scores:
            return {
                'median_sentiment_strength': 0.0,
                'median_sentiment_polarization': 0.0,
                'mean_sentiment_strength': 0.0,
                'mean_sentiment_polarization': 0.0
            }
        
        compounds = np.array(compound_scores)
        
        # 1. Median Sentiment Strength (median absolute sentiment)
        median_sentiment_strength = np.median(np.abs(compounds))
        
        # 2. Median Sentiment Polarization (median absolute deviation from neutral)
        abs_deviations_from_neutral = np.abs(compounds - 0.0)
        median_sentiment_polarization = np.median(abs_deviations_from_neutral)
        
        # 3. Mean Sentiment Strength (mean absolute sentiment)
        mean_sentiment_strength = np.mean(np.abs(compounds))
        
        # 4. Mean Sentiment Polarization (mean absolute deviation from neutral)
        mean_sentiment_polarization = np.mean(abs_deviations_from_neutral)
        
        return {
            'median_sentiment_strength': round(median_sentiment_strength, 4),
            'median_sentiment_polarization': round(median_sentiment_polarization, 4),
            'mean_sentiment_strength': round(mean_sentiment_strength, 4),
            'mean_sentiment_polarization': round(mean_sentiment_polarization, 4)
        }

    def load_conversations(self, folder_path='json_data'):
        """Load Reddit conversation files"""
        conversations = []
        
        if not os.path.exists(folder_path):
            print(f"Folder '{folder_path}' not found.")
            return conversations
        
        json_files = [f for f in os.listdir(folder_path) 
                      if f.endswith('.json') and 'reddit_comments' in f]
        
        for json_file in json_files:
            file_path = os.path.join(folder_path, json_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                post_id = json_file.replace('_reddit_comments_with_time.json', '').replace('.json', '')
                conversation_type = "richly branching" if "hb" in post_id.lower() else "poorly branching"
                
                conversations.append({
                    'post_id': post_id,
                    'conversation_type': conversation_type,
                    'data': data
                })
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue
        
        return conversations

    def build_comment_tree(self, comments_data, post_time):
        """Build comment tree with level 4 cap"""
        def process_comment(comment_data, current_level=1):
            try:
                comment_time = datetime.strptime(comment_data["timestamp"], '%Y-%m-%d %H:%M:%S')
                
                comment = {
                    'author': comment_data['author'],
                    'body': comment_data['body'],
                    'timestamp': comment_time,
                    'level': current_level,
                    'replies': []
                }
                
                # Only process replies if we haven't reached max level
                if current_level < self.max_level and "replies" in comment_data and comment_data["replies"]:
                    for reply_data in comment_data["replies"]:
                        reply = process_comment(reply_data, current_level + 1)
                        if reply:
                            comment['replies'].append(reply)
                
                return comment
            except (ValueError, KeyError):
                return None
        
        top_level_comments = []
        for comment_data in comments_data:
            comment = process_comment(comment_data, 1)
            if comment:
                top_level_comments.append(comment)
        
        return top_level_comments

    def calculate_subtree_max_depth(self, comment):
        """Calculate the maximum depth of a subtree"""
        if not comment['replies']:
            return comment['level']
        
        max_child_depth = max(self.calculate_subtree_max_depth(reply) for reply in comment['replies'])
        return max_child_depth

    def find_deepest_subtree(self, top_level_comments):
        """Find the subtree with maximum depth (deepest TST)"""
        if not top_level_comments:
            return None, None
        
        best_comment = None
        best_depth = 0
        
        for comment in top_level_comments:
            depth = self.calculate_subtree_max_depth(comment)
            if depth > best_depth:
                best_depth = depth
                best_comment = comment
        
        return best_comment, best_depth

    def extract_all_tst_comments(self, tst_root, post_time):
        """Extract all comments from TST with level and time info"""
        all_comments = []
        
        def traverse(comment, level, post_time):
            # Stop if we exceed max_level
            if level > self.max_level:
                return
                
            time_from_post = (comment['timestamp'] - post_time).total_seconds() / 3600
            
            all_comments.append({
                'text': comment['body'],
                'author': comment['author'],
                'level': level,
                'time_hours': time_from_post,
                'timestamp': comment['timestamp']
            })
            
            # Process all replies if within level limit
            for reply in comment['replies']:
                traverse(reply, level + 1, post_time)
        
        traverse(tst_root, 1, post_time)  # TST root starts at level 1
        return all_comments

    def analyze_tst_sentiment(self, conversation_data, post_id, conversation_type):
        """Analyze sentiment within the TST - ESSENTIAL COLUMNS ONLY"""
        try:
            post_title = conversation_data.get('post_title', 'No Title')
            post_timestamp = conversation_data.get('post_timestamp', '')
            
            post_time = datetime.strptime(post_timestamp, '%Y-%m-%d %H:%M:%S')
            
            # Build tree and find TST (with level cap)
            top_level_comments = self.build_comment_tree(conversation_data["comments"], post_time)
            
            if not top_level_comments:
                return None
            
            tst_root, tst_max_depth = self.find_deepest_subtree(top_level_comments)
            if not tst_root:
                return None
            
            # Extract all TST comments
            tst_comments = self.extract_all_tst_comments(tst_root, post_time)
            
            if len(tst_comments) < 1:
                return None
            
            # Analyze sentiment for ALL TST comments
            individual_sentiment_data = []
            for i, comment in enumerate(tst_comments):
                sentiment = self.analyze_sentiment(comment['text'])
                
                sentiment_record = {
                    'comment_index': i,
                    'comment_level': comment['level'],
                    'compound_score': sentiment['compound'],
                    'positive_score': sentiment['positive'],
                    'negative_score': sentiment['negative'],
                    'neutral_score': sentiment['neutral'],
                    'sentiment_label': sentiment['sentiment_label'],
                    'comment_text': comment['text'][:150] + "..." if len(comment['text']) > 150 else comment['text'],
                    'author': comment['author'],
                    'timestamp': comment['timestamp'],
                    'time_from_post_hours': comment['time_hours']
                }
                individual_sentiment_data.append(sentiment_record)
            
            if not individual_sentiment_data:
                return None
            
            # Extract all individual values
            all_compounds = [s['compound_score'] for s in individual_sentiment_data]
            all_positives = [s['positive_score'] for s in individual_sentiment_data]
            all_negatives = [s['negative_score'] for s in individual_sentiment_data]
            all_labels = [s['sentiment_label'] for s in individual_sentiment_data]
            
            # Calculate overall sentiment and polarization scores
            overall_polarization_scores = self.calculate_polarization_scores(
                all_compounds, all_positives, all_negatives
            )
            
            # Build conversation result with ESSENTIAL COLUMNS ONLY
            result = {
                # Post ID (Essential identifier)
                'post_id': post_id,
                
                # Context & Validation Measures
                'conversation_type': conversation_type,
                'total_comments': len(tst_comments),
                'max_depth_level': tst_max_depth,
                
                # Primary Sentiment Measures (ONLY THE REQUESTED 9 METRICS)
                'median_core_sentiment_compound_score': np.median(all_compounds),
                'mean_compound_all_levels': np.mean(all_compounds),
                'sentiment_positive_ratio': sum(1 for label in all_labels if label == 'positive') / len(all_labels),
                'sentiment_negative_ratio': sum(1 for label in all_labels if label == 'negative') / len(all_labels),
                'sentiment_neutral_ratio': sum(1 for label in all_labels if label == 'neutral') / len(all_labels),
                'median_sentiment_polarization_score': overall_polarization_scores['median_sentiment_polarization'],
                'median_sentiment_strength_score': overall_polarization_scores['median_sentiment_strength'],
                'mean_sentiment_polarization': overall_polarization_scores['mean_sentiment_polarization'],
                'mean_sentiment_strength': overall_polarization_scores['mean_sentiment_strength'],
                
                # Individual sentiment data (for individual CSV)
                'individual_sentiment_data': individual_sentiment_data
            }
            
            self.results.append(result)
            return result
            
        except Exception as e:
            print(f"Error analyzing {post_id}: {e}")
            return None

    def process_all_conversations(self, folder_path='json_data'):
        """Process all conversations for streamlined TST sentiment analysis"""
        try:
            conversations = self.load_conversations(folder_path)
            
            if not conversations:
                print("No conversations loaded.")
                return
            
            print(f"Processing {len(conversations)} conversations for TST sentiment analysis...")
            print(f"Level cap: {self.max_level}")
            print("TST selection: Maximum depth subtree (deepest conversation thread)")
            print("Output: Essential sentiment columns only (9 core metrics)")
            
            for conversation in conversations:
                print(f"Processing: {conversation['post_id']} ({conversation['conversation_type']})")
                
                result = self.analyze_tst_sentiment(
                    conversation['data'], 
                    conversation['post_id'], 
                    conversation['conversation_type']
                )
                if result:
                    print(f"  TST: {result['total_comments']} comments, max depth: {result['max_depth_level']}")
                    print(f"  Median sentiment: {result['median_core_sentiment_compound_score']:.4f}")
                    print(f"  Positive ratio: {result['sentiment_positive_ratio']:.1%}")
            
            print(f"\nCompleted TST sentiment analysis of {len(self.results)} conversations")
            
        except Exception as e:
            print(f"Error processing conversations: {e}")

    def create_summary_statistics(self):
        """Create summary statistics for TST sentiment analysis"""
        if not self.results:
            print("No results to summarize")
            return None
        
        print(f"\n=== TST SENTIMENT ANALYSIS (LEVEL CAP: {self.max_level}) ===")
        print(f"Total conversations analyzed: {len(self.results)}")
        
        # Count by conversation type
        conv_type_counts = {}
        for result in self.results:
            conv_type = result['conversation_type']
            conv_type_counts[conv_type] = conv_type_counts.get(conv_type, 0) + 1
        
        print(f"Conversation type distribution: {conv_type_counts}")
        print("TST selection method: Maximum depth subtree")
        
        # Overall sentiment statistics (ESSENTIAL METRICS ONLY)
        all_medians = [result['median_core_sentiment_compound_score'] for result in self.results]
        all_means = [result['mean_compound_all_levels'] for result in self.results]
        all_positive_ratios = [result['sentiment_positive_ratio'] for result in self.results]
        all_negative_ratios = [result['sentiment_negative_ratio'] for result in self.results]
        all_neutral_ratios = [result['sentiment_neutral_ratio'] for result in self.results]
        all_median_polarizations = [result['median_sentiment_polarization_score'] for result in self.results]
        all_median_strengths = [result['median_sentiment_strength_score'] for result in self.results]
        all_mean_polarizations = [result['mean_sentiment_polarization'] for result in self.results]
        all_mean_strengths = [result['mean_sentiment_strength'] for result in self.results]
        all_max_depths = [result['max_depth_level'] for result in self.results]
        
        print(f"\nTST Sentiment Statistics (9 Core Metrics):")
        print(f"  Median core sentiment: {np.median(all_medians):.4f}")
        print(f"  Mean sentiment: {np.median(all_means):.4f}")
        print(f"  Range: {min(all_medians):.4f} to {max(all_medians):.4f}")
        print(f"  Median positive ratio: {np.median(all_positive_ratios):.1%}")
        print(f"  Median negative ratio: {np.median(all_negative_ratios):.1%}")
        print(f"  Median neutral ratio: {np.median(all_neutral_ratios):.1%}")
        print(f"  Median polarization score: {np.median(all_median_polarizations):.4f}")
        print(f"  Median strength score: {np.median(all_median_strengths):.4f}")
        print(f"  Mean polarization: {np.median(all_mean_polarizations):.4f}")
        print(f"  Mean strength: {np.median(all_mean_strengths):.4f}")
        print(f"  Average max depth: {np.mean(all_max_depths):.1f}")
        
        # By conversation type
        print(f"\nBy Conversation Type:")
        for conv_type in conv_type_counts.keys():
            type_results = [r for r in self.results if r['conversation_type'] == conv_type]
            if type_results:
                type_medians = [r['median_core_sentiment_compound_score'] for r in type_results]
                type_means = [r['mean_compound_all_levels'] for r in type_results]
                type_positive = [r['sentiment_positive_ratio'] for r in type_results]
                type_negative = [r['sentiment_negative_ratio'] for r in type_results]
                type_comments = [r['total_comments'] for r in type_results]
                type_depths = [r['max_depth_level'] for r in type_results]
                
                print(f"\n{conv_type.title()}:")
                print(f"  Count: {len(type_results)}")
                print(f"  Median core sentiment: {np.median(type_medians):.4f}")
                print(f"  Mean sentiment: {np.median(type_means):.4f}")
                print(f"  Median positive ratio: {np.median(type_positive):.1%}")
                print(f"  Median negative ratio: {np.median(type_negative):.1%}")
                print(f"  Avg TST size: {np.mean(type_comments):.1f} comments")
                print(f"  Avg max depth: {np.mean(type_depths):.1f}")
        
        return True

    def create_csv_output(self, output_folder='subtree_sentiment_analysis_level4'):
        """Create CSV output with ONLY THE REQUESTED 9 SENTIMENT METRICS"""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # 1. Conversation-level metadata (ONLY THE REQUESTED 9 METRICS)
        conversation_records = []
        for result in self.results:
            conversation_record = {
                # Post ID (Essential identifier)
                'post_id': result['post_id'],
                
                # Context & Validation Measures (keeping for reference)
                'conversation_type': result['conversation_type'],
                'total_comments': result['total_comments'],
                'max_depth_level': result['max_depth_level'],
                
                # ONLY THE REQUESTED 9 SENTIMENT METRICS
                'median_core_sentiment_compound_score': round(result['median_core_sentiment_compound_score'], 4),
                'mean_compound_all_levels': round(result['mean_compound_all_levels'], 4),
                'sentiment_positive_ratio': round(result['sentiment_positive_ratio'], 3),
                'sentiment_negative_ratio': round(result['sentiment_negative_ratio'], 3),
                'sentiment_neutral_ratio': round(result['sentiment_neutral_ratio'], 3),
                'median_sentiment_polarization_score': result['median_sentiment_polarization_score'],
                'median_sentiment_strength_score': result['median_sentiment_strength_score'],
                'mean_sentiment_polarization': result['mean_sentiment_polarization'],
                'mean_sentiment_strength': result['mean_sentiment_strength']
            }
            
            conversation_records.append(conversation_record)
        
        conversation_df = pd.DataFrame(conversation_records)
        conversation_path = os.path.join(output_folder, 'tst_sentiment_summary_level4.csv')
        conversation_df.to_csv(conversation_path, index=False)
        
        # 2. Individual sentiment values (ESSENTIAL COLUMNS ONLY)
        all_individual_records = []
        for result in self.results:
            for comment in result['individual_sentiment_data']:
                individual_record = {
                    # Post ID (Essential identifier)
                    'post_id': result['post_id'],
                    
                    # Context & Validation
                    'conversation_type': result['conversation_type'],
                    'comment_level': comment['comment_level'],
                    
                    # Primary Sentiment Measures
                    'compound_score': round(comment['compound_score'], 4),
                    'sentiment_label': comment['sentiment_label']
                }
                all_individual_records.append(individual_record)
        
        individual_df = pd.DataFrame(all_individual_records)
        individual_path = os.path.join(output_folder, 'tst_individual_sentiment_values_level4.csv')
        individual_df.to_csv(individual_path, index=False)
        
        print(f"TST CSV outputs saved to {output_folder}:")
        print(f"  - TST sentiment summary: {conversation_path}")
        print(f"  - TST individual sentiment: {individual_path}")
        print(f"  - Total individual values: {len(all_individual_records)}")
        print(f"  - Contains ONLY the requested 9 sentiment metrics")
        print(f"  - TST selection: Maximum depth subtree")
        
        return conversation_path, individual_path

    def create_visualizations(self, output_folder='subtree_sentiment_analysis_level4'):
        """Create research-appropriate visualization for TST sentiment analysis with conversation type separation"""
        if not self.results:
            print("No results to visualize")
            return
        
        # Collect data from TSTs
        all_data = []
        for result in self.results:
            for comment in result['individual_sentiment_data']:
                all_data.append({
                    'post_id': result['post_id'],
                    'conversation_type': result['conversation_type'],
                    'level': comment['comment_level'],
                    'compound': comment['compound_score'],
                    'sentiment_label': comment['sentiment_label']
                })
        
        df = pd.DataFrame(all_data)
        
        # Create research-appropriate plot with smaller dimensions
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # Smaller dimensions for research paper
        
        # Get available levels
        levels = sorted(df['level'].unique())
        conversation_types = sorted(df['conversation_type'].unique())
        
        # Plot for each conversation type
        for i, conv_type in enumerate(conversation_types):
            ax = axes[i]
            
            # Filter data for this conversation type
            type_data = df[df['conversation_type'] == conv_type]
            
            # Prepare data for box plots by level
            level_data = []
            level_labels = []
            
            for level in levels:
                level_sentiments = type_data[type_data['level'] == level]['compound'].tolist()
                if level_sentiments:  # Only add if there's data for this level
                    level_data.append(level_sentiments)
                    level_labels.append(f'L{level}')
            
            if level_data:  # Only create plot if there's data
                # Create box plots with explicit outlier settings
                bp = ax.boxplot(level_data, labels=level_labels, patch_artist=True, 
                               showfliers=True, notch=False, widths=0.6,
                               flierprops=dict(marker='o', markerfacecolor='black', 
                                             markeredgecolor='black', markersize=8, 
                                             alpha=1.0, linewidth=1))
                
                # Set all boxes to white/gray for research paper
                for patch in bp['boxes']:
                    patch.set_facecolor('white')
                    patch.set_edgecolor('black')
                    patch.set_linewidth(1.2)
                
                # Style the box plot elements with bold lines
                for element in ['whiskers', 'medians', 'caps']:
                    plt.setp(bp[element], color='black', linewidth=1.5)
                
                # Make median lines more prominent
                plt.setp(bp['medians'], color='black', linewidth=2.0)
                
                # Overlay ALL individual points as scatter plots (like original posts code)
                for j, level in enumerate([lv for lv in levels if len(type_data[type_data['level'] == lv]) > 0]):
                    level_sentiments = type_data[type_data['level'] == level]['compound']
                    
                    # Add minimal jitter to x-coordinate
                    x_jitter = np.random.normal(j+1, 0.03, len(level_sentiments))
                    
                    ax.scatter(x_jitter, level_sentiments, 
                              color='black', 
                              alpha=0.4, s=8, edgecolors='none')
            
            # Research-appropriate styling with bolder labels
            ax.set_title(f'{conv_type.title()}', fontweight='bold', fontsize=12, pad=10)
            ax.set_ylabel('VADER Compound Score', fontweight='bold', fontsize=11)
            ax.set_xlabel('TST Comment Level', fontweight='bold', fontsize=11)
            
            # Add neutral reference line
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1.0)
            
            # Minimal grid for research paper
            ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
            ax.set_ylim(-1.05, 1.05)
            
            # Bold tick labels
            ax.tick_params(axis='both', which='major', labelsize=10, width=1.2)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')
            
            # Add sample size information in research format
            total_comments = len(type_data)
            ax.text(0.98, 0.02, f'n = {total_comments}', transform=ax.transAxes, 
                    va='bottom', ha='right', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor='black', linewidth=1))
        
        # Adjust layout for research paper format
        plt.tight_layout(pad=2.0)
        
        # Remove top and right spines for cleaner look
        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.2)
            ax.spines['bottom'].set_linewidth(1.2)
        
        # Save plot to graphs folder with research-appropriate settings
        graphs_folder = 'graphs'
        if not os.path.exists(graphs_folder):
            os.makedirs(graphs_folder)
        plot_path = os.path.join(graphs_folder, f'tst_sentiment_research_level{self.max_level}_maxdepth.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', 
                    edgecolor='none', format='png')
        plt.show()
        
        print(f"Research-appropriate TST visualization saved to: {plot_path}")
        return plot_path

# Main execution function
def run_tst_sentiment_analysis(max_level=4):
    """Main function to run TST sentiment analysis with only 9 core metrics"""
    print(f"=== TST SENTIMENT ANALYSIS (LEVEL CAP: {max_level}) ===")
    print("Analyzing ONLY the 9 requested sentiment metrics within Top Subtrees (TSTs):")
    print("1. median_core_sentiment_compound_score")
    print("2. mean_compound_all_levels")
    print("3. sentiment_positive_ratio")
    print("4. sentiment_negative_ratio")
    print("5. sentiment_neutral_ratio")
    print("6. median_sentiment_polarization_score")
    print("7. median_sentiment_strength_score")
    print("8. mean_sentiment_polarization")
    print("9. mean_sentiment_strength")
    print(f"- Level cap: {max_level}")
    print("- TST identification: Maximum depth subtree (deepest conversation thread)")
    print("- UPDATED: Uses max depth instead of scaling factor formula")
    print("- Visualization: Research-appropriate box plots with no colors")
    
    # Initialize analyzer with level cap
    analyzer = TSTSentimentAnalyzer(max_level=max_level)
    
    # Process all conversations
    analyzer.process_all_conversations()
    
    # Create summary statistics
    analyzer.create_summary_statistics()
    
    # Create CSV outputs (only 9 core metrics)
    analyzer.create_csv_output()
    
    # Create visualizations
    analyzer.create_visualizations()
    
    print(f"\n=== TST SENTIMENT ANALYSIS COMPLETE ===")
    print(f"Generated files with ONLY the 9 requested sentiment metrics:")
    print(f"- tst_sentiment_summary_level4.csv")
    print(f"- tst_individual_sentiment_values_level4.csv")
    print(f"- Results in 'subtree_sentiment_analysis_level4' folder")
    print(f"- TST selection method: Maximum depth subtree")
    print(f"- Focus on sentiment within deepest conversation threads")
    print(f"- Research-appropriate visualization with no colors")
    
    return analyzer

# Run the analysis
if __name__ == "__main__":
    analyzer = run_tst_sentiment_analysis(max_level=4)