import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
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

class GeneralSentimentAnalyzer:
    """
    Analyzes general sentiment across conversation levels 1-4 (no original post tracking)
    Uses level cap instead of time window
    Preserves individual values and uses medians/ranges
    MODIFIED: Keeps only essential columns for sentiment measurement
    UPDATED: Includes research-appropriate visualization
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
        """Calculate overall sentiment strength and polarization metrics using medians and means"""
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
        # Calculate absolute deviations from 0 (neutral), then take median
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

    def extract_all_comments_recursive(self, comments_data, post_time, current_level=1):
        """Recursively extract ALL comments at ALL levels up to max_level"""
        all_comments = []
        
        # Stop if we've reached the maximum level
        if current_level > self.max_level:
            return all_comments
        
        for comment_idx, comment_data in enumerate(comments_data):
            try:
                comment_time = datetime.strptime(comment_data["timestamp"], '%Y-%m-%d %H:%M:%S')
                
                # Current level comment
                comment_info = {
                    'author': comment_data['author'],
                    'body': comment_data['body'],
                    'timestamp': comment_data['timestamp'],
                    'level': current_level,
                    'comment_id': f"L{current_level}_{comment_idx}",
                    'has_replies': bool(comment_data.get('replies')),
                    'time_from_post': (comment_time - post_time).total_seconds() / 3600  # hours
                }
                all_comments.append(comment_info)
                
                # Recursively process replies if they exist and we haven't reached max level
                if comment_data.get('replies') and current_level < self.max_level:
                    child_comments = self.extract_all_comments_recursive(
                        comment_data['replies'], 
                        post_time, 
                        current_level + 1
                    )
                    all_comments.extend(child_comments)
                        
            except (ValueError, KeyError) as e:
                print(f"Error processing comment at level {current_level}: {e}")
                continue
                
        return all_comments

    def analyze_conversation_sentiment(self, conversation_data, post_id, conversation_type):
        """Analyze general sentiment across all levels (no post tracking)"""
        try:
            # Extract basic info
            post_title = conversation_data.get('post_title', 'No Title')
            post_timestamp = conversation_data.get('post_timestamp', '')
            
            # Parse timestamps
            post_time = datetime.strptime(post_timestamp, '%Y-%m-%d %H:%M:%S')
            
            # Extract ALL comments recursively up to max_level
            all_comments = self.extract_all_comments_recursive(
                conversation_data["comments"], post_time
            )
            
            if not all_comments:
                print(f"No comments found for {post_id}")
                return None
            
            # Filter out comments beyond max_level (safety check)
            all_comments = [c for c in all_comments if c['level'] <= self.max_level]
            
            # Process ALL comments and preserve individual values
            individual_sentiment_data = []
            
            for i, comment in enumerate(all_comments):
                sentiment = self.analyze_sentiment(comment['body'])
                
                sentiment_record = {
                    'comment_index': i,
                    'comment_id': comment['comment_id'],
                    'comment_level': comment['level'],
                    'compound_score': sentiment['compound'],
                    'positive_score': sentiment['positive'],
                    'negative_score': sentiment['negative'],
                    'neutral_score': sentiment['neutral'],
                    'sentiment_label': sentiment['sentiment_label'],
                    'comment_text': comment['body'][:150] + "..." if len(comment['body']) > 150 else comment['body'],
                    'author': comment['author'],
                    'timestamp': comment['timestamp'],
                    'time_from_post_hours': comment['time_from_post'],
                    'has_replies': comment['has_replies']
                }
                individual_sentiment_data.append(sentiment_record)
            
            if not individual_sentiment_data:
                print(f"No valid sentiment calculations for {post_id}")
                return None
            
            # Extract all individual values
            all_compounds = [s['compound_score'] for s in individual_sentiment_data]
            all_positives = [s['positive_score'] for s in individual_sentiment_data]
            all_negatives = [s['negative_score'] for s in individual_sentiment_data]
            all_neutrals = [s['neutral_score'] for s in individual_sentiment_data]
            all_labels = [s['sentiment_label'] for s in individual_sentiment_data]
            
            # Calculate overall sentiment and polarization scores
            overall_polarization_scores = self.calculate_polarization_scores(
                all_compounds, all_positives, all_negatives
            )
            
            # Group by level for level-specific analysis (ESSENTIAL DATA ONLY)
            level_data = {}
            max_level = max(s['comment_level'] for s in individual_sentiment_data)
            min_level = min(s['comment_level'] for s in individual_sentiment_data)
            
            for level in range(min_level, max_level + 1):
                level_comments = [s for s in individual_sentiment_data if s['comment_level'] == level]
                if level_comments:
                    level_compounds = [s['compound_score'] for s in level_comments]
                    level_labels = [s['sentiment_label'] for s in level_comments]
                    
                    level_data[level] = {
                        'median_compound': np.median(level_compounds),
                        'sentiment_positive_ratio': sum(1 for label in level_labels if label == 'positive') / len(level_labels)
                    }
            
            # Build conversation result with ESSENTIAL COLUMNS ONLY
            conversation_result = {
                # Post ID (Essential identifier)
                'post_id': post_id,
                
                # Context & Validation Measures
                'conversation_type': conversation_type,
                'total_comments': len(all_comments),
                'max_depth_level': max_level,
                
                # Primary Sentiment Measures
                'median_compound_all_levels': np.median(all_compounds),
                'mean_compound_all_levels': np.mean(all_compounds),
                'sentiment_positive_ratio': sum(1 for label in all_labels if label == 'positive') / len(all_labels),
                'sentiment_negative_ratio': sum(1 for label in all_labels if label == 'negative') / len(all_labels),
                'sentiment_neutral_ratio': sum(1 for label in all_labels if label == 'neutral') / len(all_labels),
                'median_sentiment_polarization': overall_polarization_scores['median_sentiment_polarization'],
                'median_sentiment_strength': overall_polarization_scores['median_sentiment_strength'],
                'mean_sentiment_polarization': overall_polarization_scores['mean_sentiment_polarization'],
                'mean_sentiment_strength': overall_polarization_scores['mean_sentiment_strength'],
                
                # Level-specific essential data
                'level_sentiment_data': level_data,
                
                # Individual sentiment data (for individual CSV)
                'individual_sentiment_data': individual_sentiment_data
            }
            
            self.results.append(conversation_result)
            return conversation_result
            
        except Exception as e:
            print(f"Error analyzing conversation {post_id}: {e}")
            return None

    def process_all_conversations(self, folder_path='json_data'):
        """Process all conversations for general sentiment analysis"""
        try:
            conversations = self.load_conversations(folder_path)
            
            if not conversations:
                print("No conversations loaded.")
                return
            
            print(f"Processing {len(conversations)} conversations for essential sentiment analysis...")
            print(f"Level cap: {self.max_level}")
            
            for conversation in conversations:
                print(f"Processing: {conversation['post_id']} ({conversation['conversation_type']})")
                
                result = self.analyze_conversation_sentiment(
                    conversation['data'], 
                    conversation['post_id'], 
                    conversation['conversation_type']
                )
                if result:
                    print(f"  {result['total_comments']} comments (max depth: {result['max_depth_level']})")
                    print(f"  Median sentiment: {result['median_compound_all_levels']:.4f}")
                    print(f"  Positive ratio: {result['sentiment_positive_ratio']:.1%}")
            
            print(f"\nCompleted analysis of {len(self.results)} conversations")
            
        except Exception as e:
            print(f"Error processing conversations: {e}")

    def create_summary_statistics(self):
        """Create summary statistics for essential sentiment analysis"""
        if not self.results:
            print("No results to summarize")
            return None
        
        print(f"\n=== ESSENTIAL SENTIMENT ANALYSIS (LEVEL CAP: {self.max_level}) ===")
        print(f"Total conversations analyzed: {len(self.results)}")
        
        # Count by conversation type
        conv_type_counts = {}
        for result in self.results:
            conv_type = result['conversation_type']
            conv_type_counts[conv_type] = conv_type_counts.get(conv_type, 0) + 1
        
        print(f"Conversation type distribution: {conv_type_counts}")
        
        # Overall sentiment statistics (ESSENTIAL METRICS ONLY)
        all_medians = [result['median_compound_all_levels'] for result in self.results]
        all_positive_ratios = [result['sentiment_positive_ratio'] for result in self.results]
        all_negative_ratios = [result['sentiment_negative_ratio'] for result in self.results]
        all_polarizations = [result['median_sentiment_polarization'] for result in self.results]
        all_strengths = [result['median_sentiment_strength'] for result in self.results]
        
        all_mean_strengths = [result['mean_sentiment_strength'] for result in self.results]
        all_mean_polarizations = [result['mean_sentiment_polarization'] for result in self.results]
        all_mean_compounds = [result['mean_compound_all_levels'] for result in self.results]
        
        print(f"\nEssential Sentiment Statistics:")
        print(f"  Median sentiment (all conversations): {np.median(all_medians):.4f}")
        print(f"  Mean sentiment (all conversations): {np.median(all_mean_compounds):.4f}")
        print(f"  Range: {min(all_medians):.4f} to {max(all_medians):.4f}")
        print(f"  Median positive ratio: {np.median(all_positive_ratios):.1%}")
        print(f"  Median negative ratio: {np.median(all_negative_ratios):.1%}")
        print(f"  Median polarization: {np.median(all_polarizations):.4f}")
        print(f"  Mean polarization: {np.median(all_mean_polarizations):.4f}")
        print(f"  Median sentiment strength: {np.median(all_strengths):.4f}")
        print(f"  Mean sentiment strength: {np.median(all_mean_strengths):.4f}")
        
        # By conversation type
        print(f"\nBy Conversation Type:")
        for conv_type in conv_type_counts.keys():
            type_results = [r for r in self.results if r['conversation_type'] == conv_type]
            if type_results:
                type_medians = [r['median_compound_all_levels'] for r in type_results]
                type_positive = [r['sentiment_positive_ratio'] for r in type_results]
                type_negative = [r['sentiment_negative_ratio'] for r in type_results]
                type_polarizations = [r['median_sentiment_polarization'] for r in type_results]
                type_mean_compounds = [r['mean_compound_all_levels'] for r in type_results]
                type_mean_polarizations = [r['mean_sentiment_polarization'] for r in type_results]
                type_mean_strengths = [r['mean_sentiment_strength'] for r in type_results]
                type_comments = [r['total_comments'] for r in type_results]
                
                print(f"\n{conv_type.title()}:")
                print(f"  Count: {len(type_results)}")
                print(f"  Median sentiment: {np.median(type_medians):.4f}")
                print(f"  Mean sentiment: {np.median(type_mean_compounds):.4f}")
                print(f"  Median positive ratio: {np.median(type_positive):.1%}")
                print(f"  Median negative ratio: {np.median(type_negative):.1%}")
                print(f"  Median polarization: {np.median(type_polarizations):.4f}")
                print(f"  Mean polarization: {np.median(type_mean_polarizations):.4f}")
                print(f"  Avg comment count: {np.mean(type_comments):.1f}")
        
        return True

    def create_csv_output(self, output_folder='original_posts_sentiment_analysis_level4'):
        """Create CSV output with ESSENTIAL COLUMNS ONLY - using original filenames"""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # 1. Conversation-level metadata (ESSENTIAL COLUMNS ONLY)
        conversation_records = []
        for result in self.results:
            conversation_record = {
                # Post ID (Essential identifier)
                'post_id': result['post_id'],
                
                # Context & Validation Measures
                'conversation_type': result['conversation_type'],
                'total_comments': result['total_comments'],
                'max_depth_level': result['max_depth_level'],
                
                # Primary Sentiment Measures - RENAMED COLUMNS
                'median_core_sentiment_compound_score': round(result['median_compound_all_levels'], 4),
                'mean_compound_all_levels': round(result['mean_compound_all_levels'], 4),
                'sentiment_positive_ratio': round(result['sentiment_positive_ratio'], 3),
                'sentiment_negative_ratio': round(result['sentiment_negative_ratio'], 3),
                'sentiment_neutral_ratio': round(result['sentiment_neutral_ratio'], 3),
                'median_sentiment_polarization_score': result['median_sentiment_polarization'],
                'median_sentiment_strength_score': result['median_sentiment_strength'],
                'mean_sentiment_polarization': result['mean_sentiment_polarization'],
                'mean_sentiment_strength': result['mean_sentiment_strength']
            }
            
            # Add level-specific essential data
            for level, level_data in result['level_sentiment_data'].items():
                conversation_record[f'level_{level}_median_compound'] = round(level_data['median_compound'], 4)
                conversation_record[f'level_{level}_sentiment_positive_ratio'] = round(level_data['sentiment_positive_ratio'], 3)
            
            conversation_records.append(conversation_record)
        
        conversation_df = pd.DataFrame(conversation_records)
        conversation_path = os.path.join(output_folder, 'conversation_original_posts_sentiment_level4.csv')
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
        individual_path = os.path.join(output_folder, 'all_individual_sentiment_values_original_posts_level4.csv')
        individual_df.to_csv(individual_path, index=False)
        
        print(f"Essential CSV outputs saved to {output_folder}:")
        print(f"  - Streamlined conversation sentiment: {conversation_path}")
        print(f"  - Streamlined individual sentiment: {individual_path}")
        print(f"  - Total individual values: {len(all_individual_records)}")
        
        return conversation_path, individual_path

    def create_research_visualizations(self, output_folder='original_posts_sentiment_analysis_level4'):
        """Create research-appropriate visualizations - black and white, smaller size, bold labels"""
        if not self.results:
            print("No results to visualize")
            return
        
        # Collect essential data only
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
        
        # Create research-appropriate plot (smaller, black and white)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # Smaller figure size for research paper
        
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
                # Create box plots (no colors, black and white only)
                bp = ax.boxplot(level_data, labels=level_labels, patch_artist=True, 
                               showfliers=True, notch=False, widths=0.6)
                
                # Set all boxes to white with black borders
                for patch in bp['boxes']:
                    patch.set_facecolor('white')
                    patch.set_edgecolor('black')
                    patch.set_linewidth(1.5)
                
                # Style the box plot elements (all black, bolder lines)
                for element in ['whiskers', 'fliers', 'medians', 'caps']:
                    plt.setp(bp[element], color='black', linewidth=1.5)
                
                # Make medians more prominent
                plt.setp(bp['medians'], color='black', linewidth=2.5)
                
                # Overlay individual points (black dots, minimal jitter)
                for j, level in enumerate([lv for lv in levels if len(type_data[type_data['level'] == lv]) > 0]):
                    level_sentiments = type_data[type_data['level'] == level]['compound']
                    
                    # Add minimal jitter to x-coordinate
                    x_jitter = np.random.normal(j+1, 0.03, len(level_sentiments))
                    
                    ax.scatter(x_jitter, level_sentiments, 
                              color='black', 
                              alpha=0.4, s=8, edgecolors='none')
            
            # Bold, research-appropriate styling
            title_text = f'{conv_type.replace("richly branching", "High Branching").replace("poorly branching", "Low Branching")}'
            ax.set_title(title_text, fontweight='bold', fontsize=12, pad=10)
            ax.set_ylabel('VADER Compound Score', fontweight='bold', fontsize=11)
            ax.set_xlabel('Comment Level', fontweight='bold', fontsize=11)
            
            # Neutral line (gray, thinner)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
            
            # Minimal grid
            ax.grid(True, alpha=0.2, linewidth=0.5)
            ax.set_ylim(-1.05, 1.05)
            
            # Bold tick labels
            ax.tick_params(axis='both', which='major', labelsize=10, width=1.2)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')
            
            # Add sample size information (smaller, cleaner)
            total_comments = len(type_data)
            ax.text(0.95, 0.05, f'n = {total_comments}', transform=ax.transAxes, 
                    va='bottom', ha='right', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor='black', linewidth=1))
        
        # Adjust layout for research paper
        plt.tight_layout(pad=2.0)
        
        # Remove top and right spines for cleaner look
        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.2)
            ax.spines['bottom'].set_linewidth(1.2)
        
        # Save plot to graphs folder (high DPI for research quality)
        graphs_folder = 'graphs'
        if not os.path.exists(graphs_folder):
            os.makedirs(graphs_folder)
        plot_path = os.path.join(graphs_folder, f'research_sentiment_analysis_level{self.max_level}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        
        # Also save as PDF for research papers
        pdf_path = os.path.join(graphs_folder, f'research_sentiment_analysis_level{self.max_level}.pdf')
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')
        
        plt.show()
        
        print(f"Research-appropriate visualization saved to:")
        print(f"  PNG: {plot_path}")
        print(f"  PDF: {pdf_path}")
        print(f"Features: Black & white, bold labels, smaller size, clean design")
        return plot_path, pdf_path

    def create_visualizations(self, output_folder='original_posts_sentiment_analysis_level4'):
        """Create original colored visualizations for comparison"""
        if not self.results:
            print("No results to visualize")
            return
        
        # Collect essential data only
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
        
        # Create plot with conversation type separation
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Get available levels
        levels = sorted(df['level'].unique())
        conversation_types = sorted(df['conversation_type'].unique())
        
        # Color schemes for each conversation type
        colors = {
            'richly branching': plt.cm.viridis(np.linspace(0, 0.8, len(levels))),
            'poorly branching': plt.cm.plasma(np.linspace(0, 0.8, len(levels)))
        }
        
        # Plot for each conversation type
        for i, conv_type in enumerate(conversation_types):
            ax = axes[i]
            
            # Filter data for this conversation type
            type_data = df[df['conversation_type'] == conv_type]
            
            # Prepare data for box plots by level
            level_data = []
            level_labels = []
            level_colors = []
            
            for level in levels:
                level_sentiments = type_data[type_data['level'] == level]['compound'].tolist()
                if level_sentiments:  # Only add if there's data for this level
                    level_data.append(level_sentiments)
                    level_labels.append(f'L{level}')
                    level_colors.append(colors[conv_type][level-1])
            
            if level_data:  # Only create plot if there's data
                # Create box plots
                bp = ax.boxplot(level_data, labels=level_labels, patch_artist=True, 
                               showfliers=True, notch=False, widths=0.6)
                
                # Color the boxes
                for patch, color in zip(bp['boxes'], level_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                # Style the box plot elements
                for element in ['whiskers', 'fliers', 'medians', 'caps']:
                    plt.setp(bp[element], color='black', linewidth=1.2)
                
                # Overlay individual points with jitter
                for j, level in enumerate([lv for lv in levels if len(type_data[type_data['level'] == lv]) > 0]):
                    level_sentiments = type_data[type_data['level'] == level]['compound']
                    
                    # Add jitter to x-coordinate to spread points horizontally
                    x_jitter = np.random.normal(j+1, 0.05, len(level_sentiments))
                    
                    ax.scatter(x_jitter, level_sentiments, 
                              color=level_colors[j], 
                              alpha=0.6, s=15, edgecolors='white', linewidth=0.5)
            
            # Styling
            ax.set_title(f'Original Posts Sentiment Analysis by Level\n{conv_type.title()}', fontweight='bold', pad=15)
            ax.set_ylabel('VADER Compound Score', fontweight='bold')
            ax.set_xlabel('Comment Level', fontweight='bold')
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Neutral')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-1.05, 1.05)
            
            # Add sentiment scale labels
            ax.text(0.02, 0.98, 'Positive', transform=ax.transAxes, va='top', 
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            ax.text(0.02, 0.02, 'Negative', transform=ax.transAxes, va='bottom',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
            
            # Add sample size information
            total_comments = len(type_data)
            ax.text(0.98, 0.02, f'n = {total_comments} comments', transform=ax.transAxes, 
                    va='bottom', ha='right', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot to graphs folder
        graphs_folder = 'graphs'
        if not os.path.exists(graphs_folder):
            os.makedirs(graphs_folder)
        plot_path = os.path.join(graphs_folder, f'original_posts_sentiment_by_conversation_type_level{self.max_level}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Original colored visualization saved to: {plot_path}")
        return plot_path


# Main execution function
def run_essential_sentiment_analysis(max_level=4):
    """Main function to run essential sentiment analysis with original visualization"""
    print(f"=== STREAMLINED SENTIMENT ANALYSIS (LEVEL CAP: {max_level}) ===")
    print("Analyzing ESSENTIAL sentiment metrics only:")
    print("- Post ID (identifier)")
    print("- Context & Validation: conversation_type, total_comments, max_depth_level")
    print("- Primary Sentiment: median_core_sentiment_compound_score, mean_compound, ratios, polarization_score, strength_score")
    print("- Level-specific: median_compound and sentiment_positive_ratio by level")
    print(f"- Level cap: {max_level}")
    print("- REPLACES original files with streamlined essential columns")
    print("- Now includes both median and mean-based metrics")
    print("- Updated column names for better clarity")
    print("- UPDATED: Visualization now separates Richly vs Poorly Branching conversation types")
    print("- UPDATED: sentiment ratio columns renamed to sentiment_positive_ratio, sentiment_negative_ratio, sentiment_neutral_ratio")
    
    # Initialize analyzer with level cap
    analyzer = GeneralSentimentAnalyzer(max_level=max_level)
    
    # Process all conversations
    analyzer.process_all_conversations()
    
    # Create summary statistics
    analyzer.create_summary_statistics()
    
    # Create CSV outputs (essential columns only, original filenames)
    analyzer.create_csv_output()
    
    # Create original colored visualizations
    analyzer.create_visualizations()
    
    print(f"\n=== STREAMLINED SENTIMENT ANALYSIS COMPLETE ===")
    print(f"REPLACED original files with essential sentiment metrics:")
    print(f"- conversation_original_posts_sentiment_level4.csv (streamlined)")
    print(f"- all_individual_sentiment_values_original_posts_level4.csv (streamlined)")
    print(f"- Results in 'original_posts_sentiment_analysis_level4' folder")
    print(f"- Clean, focused output with only the most important sentiment columns")  
    print(f"- Now includes both median and mean-based sentiment metrics for comprehensive analysis")
    print(f"- UPDATED: Visualization shows Richly vs Poorly Branching conversation types separately")
    
    return analyzer


# Research-specific execution function
def run_research_sentiment_analysis(max_level=4):
    """Main function to run essential sentiment analysis with research-appropriate visualization"""
    print(f"=== RESEARCH SENTIMENT ANALYSIS (LEVEL CAP: {max_level}) ===")
    print("Creating research-appropriate visualization:")
    print("- Black and white only (no colors)")
    print("- Bold, clear labels")
    print("- Smaller figure size for papers")
    print("- High DPI output (PNG + PDF)")
    print("- Clean, minimal design")
    
    # Initialize analyzer with level cap
    analyzer = GeneralSentimentAnalyzer(max_level=max_level)
    
    # Process all conversations
    analyzer.process_all_conversations()
    
    # Create summary statistics
    analyzer.create_summary_statistics()
    
    # Create CSV outputs
    analyzer.create_csv_output()
    
    # Create research-appropriate visualizations
    analyzer.create_research_visualizations()
    
    print(f"\n=== RESEARCH VISUALIZATION COMPLETE ===")
    print(f"Output files:")
    print(f"- research_sentiment_analysis_level4.png (300 DPI)")
    print(f"- research_sentiment_analysis_level4.pdf (vector format)")
    print(f"- Ready for research paper inclusion")
    
    return analyzer


# Combined execution function
def run_complete_sentiment_analysis(max_level=4):
    """Run both original and research visualizations"""
    print(f"=== COMPLETE SENTIMENT ANALYSIS (LEVEL CAP: {max_level}) ===")
    print("Running full analysis with both visualization types:")
    print("1. Original colored visualization")
    print("2. Research-appropriate black & white visualization")
    
    # Initialize analyzer with level cap
    analyzer = GeneralSentimentAnalyzer(max_level=max_level)
    
    # Process all conversations
    analyzer.process_all_conversations()
    
    # Create summary statistics
    analyzer.create_summary_statistics()
    
    # Create CSV outputs
    analyzer.create_csv_output()
    
    # Create both types of visualizations
    print("\nCreating original colored visualization...")
    analyzer.create_visualizations()
    
    print("\nCreating research-appropriate visualization...")
    analyzer.create_research_visualizations()
    
    print(f"\n=== COMPLETE ANALYSIS FINISHED ===")
    print(f"Generated files:")
    print(f"- conversation_original_posts_sentiment_level4.csv")
    print(f"- all_individual_sentiment_values_original_posts_level4.csv")
    print(f"- original_posts_sentiment_by_conversation_type_level4.png (colored)")
    print(f"- research_sentiment_analysis_level4.png (B&W, 300 DPI)")
    print(f"- research_sentiment_analysis_level4.pdf (vector)")
    
    return analyzer


# Run the analysis based on your needs
if __name__ == "__main__":
    # Choose one of these options:
    
    # Option 1: Original analysis with colored visualization
    # analyzer = run_essential_sentiment_analysis(max_level=4)
    
    # Option 2: Research-only analysis with B&W visualization
    analyzer = run_research_sentiment_analysis(max_level=4)
    
    # Option 3: Complete analysis with both visualizations
    # analyzer = run_complete_sentiment_analysis(max_l