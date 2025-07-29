import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load both CSV files and prepare data for correlation analysis"""
    
    # Load the datasets
    original_df = pd.read_csv('original_posts_sentiment_analysis_level4/all_individual_sentiment_values_original_posts_level4.csv')
    tst_df = pd.read_csv('subtree_sentiment_analysis_level4/tst_individual_sentiment_values_level4.csv')
    
    print("Original Posts Dataset:")
    print(f"  Total comments: {len(original_df)}")
    print(f"  Unique posts: {original_df['post_id'].nunique()}")
    print(f"  Conversation types: {original_df['conversation_type'].value_counts().to_dict()}")
    
    print("\nTST Subtrees Dataset:")
    print(f"  Total comments: {len(tst_df)}")
    print(f"  Unique posts: {tst_df['post_id'].nunique()}")
    print(f"  Conversation types: {tst_df['conversation_type'].value_counts().to_dict()}")
    
    return original_df, tst_df

def aggregate_post_level_sentiment(df, dataset_name):
    """Aggregate sentiment scores at the post level"""
    
    post_aggregated = df.groupby(['post_id', 'conversation_type']).agg({
        'compound_score': ['mean', 'median', 'std', 'count']
    }).round(4)
    
    # Flatten column names
    post_aggregated.columns = ['mean_compound', 'median_compound', 'std_compound', 'comment_count']
    post_aggregated = post_aggregated.reset_index()
    
    print(f"\n{dataset_name} - Post-level aggregation:")
    print(f"  Posts: {len(post_aggregated)}")
    print(f"  Avg comments per post: {post_aggregated['comment_count'].mean():.1f}")
    
    return post_aggregated

def create_correlation_analysis(original_agg, tst_agg, conversation_type, metric='mean_compound'):
    """Create correlation analysis for a specific conversation type"""
    
    # Filter by conversation type
    orig_filtered = original_agg[original_agg['conversation_type'] == conversation_type].copy()
    tst_filtered = tst_agg[tst_agg['conversation_type'] == conversation_type].copy()
    
    # Merge on post_id to get matched pairs
    merged = pd.merge(orig_filtered[['post_id', metric]], 
                     tst_filtered[['post_id', metric]], 
                     on='post_id', 
                     suffixes=('_original', '_tst'))
    
    if len(merged) == 0:
        print(f"No matching posts found for {conversation_type}")
        return None
    
    # Calculate correlation statistics
    correlation, p_value = stats.pearsonr(merged[f'{metric}_original'], merged[f'{metric}_tst'])
    r_squared = r2_score(merged[f'{metric}_original'], merged[f'{metric}_tst'])
    
    # Calculate additional statistics
    n_posts = len(merged)
    
    print(f"\n{conversation_type.title()} Conversations:")
    print(f"  Matched posts: {n_posts}")
    print(f"  Correlation (r): {correlation:.4f}")
    print(f"  P-value: {p_value:.2e}")
    print(f"  R-squared: {r_squared:.4f}")
    print(f"  Significance: {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
    
    return {
        'data': merged,
        'correlation': correlation,
        'p_value': p_value,
        'r_squared': r_squared,
        'n_posts': n_posts,
        'conversation_type': conversation_type
    }

def create_correlation_plots(richly_results, poorly_results, metric='mean_compound'):
    """Create side-by-side correlation plots for research paper"""
    
    # Research paper appropriate dimensions (smaller)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    
    results_list = [richly_results, poorly_results]
    
    for i, results in enumerate(results_list):
        if results is None:
            continue
            
        ax = axes[i]
        data = results['data']
        conv_type = results['conversation_type']
        
        # Create scatter plot with no colors (black/gray for research papers)
        ax.scatter(data[f'{metric}_original'], data[f'{metric}_tst'], 
                  alpha=0.7, s=40, color='black', 
                  edgecolors='gray', linewidth=0.5)
        
        # Add perfect correlation line (y=x)
        min_val = min(data[f'{metric}_original'].min(), data[f'{metric}_tst'].min())
        max_val = max(data[f'{metric}_original'].max(), data[f'{metric}_tst'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 
               'k--', alpha=0.6, linewidth=1.5)
        
        # Add regression line
        z = np.polyfit(data[f'{metric}_original'], data[f'{metric}_tst'], 1)
        p = np.poly1d(z)
        ax.plot(data[f'{metric}_original'], p(data[f'{metric}_original']), 
               'k-', alpha=0.8, linewidth=2)
        
        # Bolder labels for research paper
        ax.set_xlabel('Original Posts Mean Compound Score', fontweight='bold', fontsize=11)
        ax.set_ylabel('TST Subtrees Mean Compound Score', fontweight='bold', fontsize=11)
        ax.set_title(f'{conv_type.title()} Conversations', 
                    fontweight='bold', fontsize=12, pad=10)
        
        # Add statistics text box (more compact for research paper)
        stats_text = f'n = {results["n_posts"]}\n'
        stats_text += f'r = {results["correlation"]:.3f}\n'
        stats_text += f'p = {results["p_value"]:.2e}\n'
        stats_text += f'R² = {results["r_squared"]:.3f}'
        
        # Determine significance stars
        if results['p_value'] < 0.001:
            sig_text = '***'
        elif results['p_value'] < 0.01:
            sig_text = '**'
        elif results['p_value'] < 0.05:
            sig_text = '*'
        else:
            sig_text = 'ns'
        
        # Position stats box more appropriately for smaller plot
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor='black', alpha=0.9, linewidth=0.5))
        
        # Significance indicator (smaller and more subtle)
        ax.text(0.95, 0.05, sig_text, transform=ax.transAxes, 
               verticalalignment='bottom', horizontalalignment='right', 
               fontsize=14, fontweight='bold', color='black')
        
        # Add subtle grid
        ax.grid(True, alpha=0.3, color='gray', linewidth=0.5)
        
        # Set equal aspect ratio and limits
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(min_val - 0.05, max_val + 0.05)
        ax.set_ylim(min_val - 0.05, max_val + 0.05)
        
        # Bold tick labels
        ax.tick_params(axis='both', which='major', labelsize=9, width=1)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
    
    # Adjust layout for research paper format
    plt.tight_layout(pad=2.0)
    
    # Save with high DPI for publication quality
    plt.savefig('correlation_analysis_original_vs_tst_research.png', 
                dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none')
    plt.show()
    
    return fig

def create_summary_statistics(richly_results, poorly_results):
    """Create summary statistics comparison"""
    
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS SUMMARY")
    print("="*60)
    print("Comparing Original Posts vs TST Subtrees Sentiment")
    print("Metric: Mean Compound Score per Post")
    print("-"*60)
    
    if richly_results:
        print(f"RICHLY BRANCHING CONVERSATIONS:")
        print(f"  Sample size: {richly_results['n_posts']} matched posts")
        print(f"  Correlation (r): {richly_results['correlation']:.4f}")
        print(f"  P-value: {richly_results['p_value']:.2e}")
        print(f"  R-squared: {richly_results['r_squared']:.4f}")
        print(f"  Effect size: {'Large' if abs(richly_results['correlation']) > 0.5 else 'Medium' if abs(richly_results['correlation']) > 0.3 else 'Small'}")
    
    print()
    
    if poorly_results:
        print(f"POORLY BRANCHING CONVERSATIONS:")
        print(f"  Sample size: {poorly_results['n_posts']} matched posts")
        print(f"  Correlation (r): {poorly_results['correlation']:.4f}")
        print(f"  P-value: {poorly_results['p_value']:.2e}")
        print(f"  R-squared: {poorly_results['r_squared']:.4f}")
        print(f"  Effect size: {'Large' if abs(poorly_results['correlation']) > 0.5 else 'Medium' if abs(poorly_results['correlation']) > 0.3 else 'Small'}")
    
    print("\n" + "-"*60)
    print("INTERPRETATION:")
    
    if richly_results and poorly_results:
        if richly_results['correlation'] > poorly_results['correlation']:
            print("• Richly branching conversations show STRONGER correlation")
            print("  between original posts and TST subtrees")
        else:
            print("• Poorly branching conversations show STRONGER correlation")
            print("  between original posts and TST subtrees")
        
        print(f"• Difference in correlation: {abs(richly_results['correlation'] - poorly_results['correlation']):.3f}")
    
    print("\nStatistical significance levels:")
    print("*** p < 0.001 (highly significant)")
    print("**  p < 0.01  (very significant)")  
    print("*   p < 0.05  (significant)")
    print("ns  p ≥ 0.05  (not significant)")

def main():
    """Main function to run the correlation analysis"""
    
    print("CORRELATION ANALYSIS: Original Posts vs TST Subtrees")
    print("="*55)
    
    # Load and prepare data
    original_df, tst_df = load_and_prepare_data()
    
    # Aggregate to post level
    original_agg = aggregate_post_level_sentiment(original_df, "Original Posts")
    tst_agg = aggregate_post_level_sentiment(tst_df, "TST Subtrees")
    
    # Perform correlation analysis for each conversation type
    richly_results = create_correlation_analysis(original_agg, tst_agg, 'richly branching')
    poorly_results = create_correlation_analysis(original_agg, tst_agg, 'poorly branching')
    
    # Create visualization
    create_correlation_plots(richly_results, poorly_results)
    
    # Print summary statistics
    create_summary_statistics(richly_results, poorly_results)
    
    print(f"\nAnalysis complete! Correlation plot saved as 'correlation_analysis_original_vs_tst_research.png'")

if __name__ == "__main__":
    main()