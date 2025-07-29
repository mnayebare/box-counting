import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

def define_post_groups():
    """Define the specific post ID groups for analysis"""
    return {
        'original_highest': ['post50hb', 'post5hb', 'post2hb', 'post57hb', 'post18hb', 
                           'post29hb', 'post1hb', 'post47hb', 'post34hb', 'post33hb'],
        'original_lowest': ['post7lb', 'post10lb', 'post21lb', 'post22lb', 'post51lb', 
                          'post20lb', 'post47lb', 'post25lb', 'post54lb', 'post5lb'],
        'subtree_highest': ['post2hb', 'post3hb', 'post1hb', 'post47hb', 'post54hb', 
                          'post5hb', 'post18hb', 'post20hb', 'post14hb', 'post28hb'],
        'subtree_lowest': ['post7lb', 'post16lb', 'post28lb', 'post12lb', 'post23lb', 
                         'post10lb', 'post21lb', 'post22lb', 'post25lb', 'post26lb']
    }

def load_and_prepare_data():
    """Load both CSV files and prepare data for correlation analysis"""
    
    # Load the datasets
    original_df = pd.read_csv('original_posts_sentiment_analysis_level4/all_individual_sentiment_values_original_posts_level4.csv')
    tst_df = pd.read_csv('subtree_sentiment_analysis_level4/tst_individual_sentiment_values_level4.csv')
    
    # Get post groups
    post_groups = define_post_groups()
    
    # Get all post IDs we're interested in
    all_original_posts = post_groups['original_highest'] + post_groups['original_lowest']
    all_subtree_posts = post_groups['subtree_highest'] + post_groups['subtree_lowest']
    
    # Filter datasets to only include our specific posts
    original_filtered = original_df[original_df['post_id'].isin(all_original_posts)].copy()
    tst_filtered = tst_df[tst_df['post_id'].isin(all_subtree_posts)].copy()
    
    print("Filtered Original Posts Dataset:")
    print(f"  Total comments: {len(original_filtered)}")
    print(f"  Unique posts: {original_filtered['post_id'].nunique()}")
    print(f"  Posts found: {sorted(original_filtered['post_id'].unique())}")
    
    print("\nFiltered TST Subtrees Dataset:")
    print(f"  Total comments: {len(tst_filtered)}")
    print(f"  Unique posts: {tst_filtered['post_id'].nunique()}")
    print(f"  Posts found: {sorted(tst_filtered['post_id'].unique())}")
    
    return original_filtered, tst_filtered, post_groups

def aggregate_post_level_sentiment(df, dataset_name, post_groups):
    """Aggregate sentiment scores at the post level with group classifications"""
    
    post_aggregated = df.groupby(['post_id', 'conversation_type']).agg({
        'compound_score': ['mean', 'median', 'std', 'count']
    }).round(4)
    
    # Flatten column names
    post_aggregated.columns = ['mean_compound', 'median_compound', 'std_compound', 'comment_count']
    post_aggregated = post_aggregated.reset_index()
    
    # Add group classification
    if 'original' in dataset_name.lower():
        post_aggregated['group'] = post_aggregated['post_id'].apply(
            lambda x: 'highest' if x in post_groups['original_highest'] else 
                     'lowest' if x in post_groups['original_lowest'] else 'other'
        )
    else:
        post_aggregated['group'] = post_aggregated['post_id'].apply(
            lambda x: 'highest_fractal' if x in post_groups['subtree_highest'] else 
                     'lowest_fractal' if x in post_groups['subtree_lowest'] else 'other'
        )
    
    print(f"\n{dataset_name} - Post-level aggregation:")
    print(f"  Posts: {len(post_aggregated)}")
    print(f"  Group distribution: {post_aggregated['group'].value_counts().to_dict()}")
    print(f"  Avg comments per post: {post_aggregated['comment_count'].mean():.1f}")
    
    return post_aggregated

def analyze_group_differences(df, dataset_name):
    """Analyze sentiment differences between high/low groups"""
    
    print(f"\n{dataset_name} - GROUP COMPARISON ANALYSIS")
    print("="*50)
    
    # Get group data
    if 'original' in dataset_name.lower():
        high_group = df[df['group'] == 'highest']
        low_group = df[df['group'] == 'lowest']
        high_label, low_label = 'Highest Posts', 'Lowest Posts'
    else:
        high_group = df[df['group'] == 'highest_fractal']
        low_group = df[df['group'] == 'lowest_fractal']
        high_label, low_label = 'Highest Fractal', 'Lowest Fractal'
    
    # Calculate descriptive statistics
    print(f"\n{high_label}:")
    print(f"  N posts: {len(high_group)}")
    print(f"  Mean compound: {high_group['mean_compound'].mean():.4f} ± {high_group['mean_compound'].std():.4f}")
    print(f"  Median compound: {high_group['mean_compound'].median():.4f}")
    print(f"  Range: [{high_group['mean_compound'].min():.4f}, {high_group['mean_compound'].max():.4f}]")
    
    print(f"\n{low_label}:")
    print(f"  N posts: {len(low_group)}")
    print(f"  Mean compound: {low_group['mean_compound'].mean():.4f} ± {low_group['mean_compound'].std():.4f}")
    print(f"  Median compound: {low_group['mean_compound'].median():.4f}")
    print(f"  Range: [{low_group['mean_compound'].min():.4f}, {low_group['mean_compound'].max():.4f}]")
    
    # Statistical testing
    if len(high_group) > 0 and len(low_group) > 0:
        # T-test
        t_stat, t_pval = stats.ttest_ind(high_group['mean_compound'], low_group['mean_compound'])
        
        # Mann-Whitney U test
        u_stat, u_pval = stats.mannwhitneyu(high_group['mean_compound'], low_group['mean_compound'], 
                                           alternative='two-sided')
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(high_group)-1)*high_group['mean_compound'].var() + 
                             (len(low_group)-1)*low_group['mean_compound'].var()) / 
                            (len(high_group) + len(low_group) - 2))
        cohens_d = (high_group['mean_compound'].mean() - low_group['mean_compound'].mean()) / pooled_std
        
        print(f"\nSTATISTICAL TESTS:")
        print(f"  T-test: t = {t_stat:.4f}, p = {t_pval:.4f}")
        print(f"  Mann-Whitney U: U = {u_stat:.1f}, p = {u_pval:.4f}")
        print(f"  Cohen's d: {cohens_d:.4f}")
        print(f"  Effect size: {'Large' if abs(cohens_d) > 0.8 else 'Medium' if abs(cohens_d) > 0.5 else 'Small'}")
        
        return {
            'high_mean': high_group['mean_compound'].mean(),
            'low_mean': low_group['mean_compound'].mean(),
            't_stat': t_stat,
            't_pval': t_pval,
            'u_pval': u_pval,
            'cohens_d': cohens_d
        }
    
    return None

def create_correlation_analysis_with_groups(original_agg, tst_agg, conversation_type, metric='mean_compound'):
    """Create correlation analysis for a specific conversation type with group information"""
    
    # Filter by conversation type
    orig_filtered = original_agg[original_agg['conversation_type'] == conversation_type].copy()
    tst_filtered = tst_agg[tst_agg['conversation_type'] == conversation_type].copy()
    
    # Merge on post_id to get matched pairs
    merged = pd.merge(orig_filtered[['post_id', metric, 'group']], 
                     tst_filtered[['post_id', metric, 'group']], 
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
    
    print(f"\n{conversation_type.title()} Conversations - MATCHED POSTS:")
    print(f"  Matched posts: {n_posts}")
    print(f"  Correlation (r): {correlation:.4f}")
    print(f"  P-value: {p_value:.2e}")
    print(f"  R-squared: {r_squared:.4f}")
    print(f"  Significance: {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
    
    # Group analysis for matched posts
    print(f"  Group distribution:")
    for orig_group in merged['group_original'].unique():
        count = len(merged[merged['group_original'] == orig_group])
        print(f"    {orig_group}: {count} posts")
    
    return {
        'data': merged,
        'correlation': correlation,
        'p_value': p_value,
        'r_squared': r_squared,
        'n_posts': n_posts,
        'conversation_type': conversation_type
    }

def create_enhanced_correlation_plots(richly_results, poorly_results, metric='mean_compound'):
    """Create enhanced correlation plots showing high/low group classifications"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    results_list = [richly_results, poorly_results]
    colors = {'highest': 'red', 'lowest': 'blue', 'highest_fractal': 'red', 'lowest_fractal': 'blue'}
    
    for i, results in enumerate(results_list):
        if results is None:
            continue
            
        ax = axes[i]
        data = results['data']
        conv_type = results['conversation_type']
        
        # Create scatter plot with colors based on group
        for group in data['group_original'].unique():
            group_data = data[data['group_original'] == group]
            color = colors.get(group, 'gray')
            label = f"{group.replace('_', ' ').title()} ({len(group_data)})"
            
            ax.scatter(group_data[f'{metric}_original'], group_data[f'{metric}_tst'], 
                      alpha=0.7, s=60, color=color, label=label,
                      edgecolors='black', linewidth=0.5)
        
        # Add perfect correlation line (y=x)
        min_val = min(data[f'{metric}_original'].min(), data[f'{metric}_tst'].min())
        max_val = max(data[f'{metric}_original'].max(), data[f'{metric}_tst'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 
               'k--', alpha=0.6, linewidth=1.5, label='Perfect correlation')
        
        # Add regression line
        z = np.polyfit(data[f'{metric}_original'], data[f'{metric}_tst'], 1)
        p = np.poly1d(z)
        ax.plot(data[f'{metric}_original'], p(data[f'{metric}_original']), 
               'k-', alpha=0.8, linewidth=2, label='Regression line')
        
        # Labels and title
        ax.set_xlabel('Original Posts Mean Compound Score', fontweight='bold', fontsize=11)
        ax.set_ylabel('TST Subtrees Mean Compound Score', fontweight='bold', fontsize=11)
        ax.set_title(f'{conv_type.title()} Conversations\n(High/Low Classification)', 
                    fontweight='bold', fontsize=12, pad=15)
        
        # Add statistics text box
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
        
        # Position stats box
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor='black', alpha=0.9, linewidth=0.5))
        
        # Significance indicator
        ax.text(0.95, 0.05, sig_text, transform=ax.transAxes, 
               verticalalignment='bottom', horizontalalignment='right', 
               fontsize=14, fontweight='bold', color='black')
        
        # Add legend
        ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
        
        # Add grid
        ax.grid(True, alpha=0.3, color='gray', linewidth=0.5)
        
        # Set equal aspect ratio and limits
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(min_val - 0.05, max_val + 0.05)
        ax.set_ylim(min_val - 0.05, max_val + 0.05)
        
        # Bold tick labels
        ax.tick_params(axis='both', which='major', labelsize=9, width=1)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
    
    plt.tight_layout(pad=2.0)
    
    # Save with high DPI for publication quality
    plt.savefig('enhanced_correlation_analysis_high_low_groups.png', 
                dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none')
    plt.show()
    
    return fig

def create_comprehensive_summary(original_stats, tst_stats, richly_results, poorly_results):
    """Create comprehensive summary with group and correlation analysis"""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE SENTIMENT ANALYSIS SUMMARY")
    print("="*70)
    print("Analysis of Specific High/Low Post Classifications")
    print("-"*70)
    
    print("\n1. GROUP DIFFERENCES WITHIN DATASETS:")
    print("-"*40)
    
    if original_stats:
        direction = "higher" if original_stats['high_mean'] > original_stats['low_mean'] else "lower"
        print(f"ORIGINAL POSTS:")
        print(f"  • Highest posts have {direction} sentiment than lowest posts")
        print(f"  • Mean difference: {abs(original_stats['high_mean'] - original_stats['low_mean']):.4f}")
        print(f"  • Statistical significance: p = {original_stats['t_pval']:.4f}")
        print(f"  • Effect size (Cohen's d): {original_stats['cohens_d']:.3f}")
    
    if tst_stats:
        direction = "higher" if tst_stats['high_mean'] > tst_stats['low_mean'] else "lower"
        print(f"\nTST SUBTREES:")
        print(f"  • Highest fractal posts have {direction} sentiment than lowest fractal posts")
        print(f"  • Mean difference: {abs(tst_stats['high_mean'] - tst_stats['low_mean']):.4f}")
        print(f"  • Statistical significance: p = {tst_stats['t_pval']:.4f}")
        print(f"  • Effect size (Cohen's d): {tst_stats['cohens_d']:.3f}")
    
    print(f"\n2. CORRELATION BETWEEN ORIGINAL POSTS AND SUBTREES:")
    print("-"*50)
    
    if richly_results:
        print(f"RICHLY BRANCHING CONVERSATIONS:")
        print(f"  • Sample size: {richly_results['n_posts']} matched posts")
        print(f"  • Correlation (r): {richly_results['correlation']:.4f}")
        print(f"  • P-value: {richly_results['p_value']:.2e}")
        print(f"  • R-squared: {richly_results['r_squared']:.4f}")
        print(f"  • Relationship strength: {'Strong' if abs(richly_results['correlation']) > 0.7 else 'Moderate' if abs(richly_results['correlation']) > 0.4 else 'Weak'}")
    
    if poorly_results:
        print(f"\nPOORLY BRANCHING CONVERSATIONS:")
        print(f"  • Sample size: {poorly_results['n_posts']} matched posts")
        print(f"  • Correlation (r): {poorly_results['correlation']:.4f}")
        print(f"  • P-value: {poorly_results['p_value']:.2e}")
        print(f"  • R-squared: {poorly_results['r_squared']:.4f}")
        print(f"  • Relationship strength: {'Strong' if abs(poorly_results['correlation']) > 0.7 else 'Moderate' if abs(poorly_results['correlation']) > 0.4 else 'Weak'}")
    
    print(f"\n3. KEY FINDINGS:")
    print("-"*15)
    
    if richly_results and poorly_results:
        if richly_results['correlation'] > poorly_results['correlation']:
            print("• Richly branching conversations show STRONGER correlation")
            print("  between original posts and subtree sentiment")
        else:
            print("• Poorly branching conversations show STRONGER correlation")
            print("  between original posts and subtree sentiment")
        
        print(f"• Correlation difference: {abs(richly_results['correlation'] - poorly_results['correlation']):.3f}")
    
    print("\n• High/low classifications can be visualized in the correlation plots")
    print("• Red points = Highest group, Blue points = Lowest group")
    print("• Analysis focuses on relationship between original post sentiment")
    print("  and corresponding subtree sentiment for your specific post selections")

def main():
    """Main function to run the enhanced correlation analysis"""
    
    print("ENHANCED CORRELATION ANALYSIS: High/Low Post Classifications")
    print("="*65)
    
    # Load and prepare data
    original_df, tst_df, post_groups = load_and_prepare_data()
    
    # Aggregate to post level with group classifications
    original_agg = aggregate_post_level_sentiment(original_df, "Original Posts", post_groups)
    tst_agg = aggregate_post_level_sentiment(tst_df, "TST Subtrees", post_groups)
    
    # Analyze group differences within each dataset
    original_stats = analyze_group_differences(original_agg, "Original Posts")
    tst_stats = analyze_group_differences(tst_agg, "TST Subtrees")
    
    # Perform correlation analysis for each conversation type
    richly_results = create_correlation_analysis_with_groups(original_agg, tst_agg, 'richly branching')
    poorly_results = create_correlation_analysis_with_groups(original_agg, tst_agg, 'poorly branching')
    
    # Create enhanced visualization
    create_enhanced_correlation_plots(richly_results, poorly_results)
    
    # Print comprehensive summary
    create_comprehensive_summary(original_stats, tst_stats, richly_results, poorly_results)
    
    print(f"\nAnalysis complete! Enhanced correlation plot saved as 'enhanced_correlation_analysis_high_low_groups.png'")

if __name__ == "__main__":
    main()