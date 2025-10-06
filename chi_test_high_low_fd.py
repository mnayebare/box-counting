import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, chi2
from scipy.stats import fisher_exact

def analyze_keyword_convergence(original_file, subtree_file):
    """
    Analyze keyword convergence between original posts and subtrees using chi-square test.
    
    Parameters:
    original_file (str): Path to clustering_results.csv
    subtree_file (str): Path to subtree_clustering_results.csv
    """
    
    # Read the data
    original_df = pd.read_csv(original_file)
    subtree_df = pd.read_csv(subtree_file)
    
    print("=== KEYWORD CONVERGENCE ANALYSIS ===")
    print(f"Original data: {len(original_df)} rows")
    print(f"Subtree data: {len(subtree_df)} rows")
    
    # Extract unique keywords
    original_keywords = set(original_df['keyword'].unique())
    subtree_keywords = set(subtree_df['keyword'].unique())
    all_keywords = original_keywords.union(subtree_keywords)
    
    # Calculate overlap
    overlap = original_keywords.intersection(subtree_keywords)
    
    print(f"\nBasic Statistics:")
    print(f"- Original keywords: {len(original_keywords)}")
    print(f"- Subtree keywords: {len(subtree_keywords)}")
    print(f"- Total unique keywords: {len(all_keywords)}")
    print(f"- Actual overlap: {len(overlap)}")
    print(f"- Overlap rate: {len(overlap)/min(len(original_keywords), len(subtree_keywords))*100:.1f}%")
    
    return original_keywords, subtree_keywords, overlap, all_keywords

def chi_square_independence_test(original_keywords, subtree_keywords, all_keywords):
    """
    Perform chi-square test for independence of keyword occurrence.
    """
    
    n_original = len(original_keywords)
    n_subtree = len(subtree_keywords)
    n_total = len(all_keywords)
    n_overlap = len(original_keywords.intersection(subtree_keywords))
    
    print("\n=== CHI-SQUARE TEST FOR INDEPENDENCE ===")
    
    # Expected overlap under independence assumption
    p_original = n_original / n_total
    p_subtree = n_subtree / n_total
    expected_overlap = n_total * p_original * p_subtree
    
    print(f"Expected overlap (random): {expected_overlap:.2f}")
    print(f"Observed overlap: {n_overlap}")
    print(f"Difference: {n_overlap - expected_overlap:.2f}")
    
    # Chi-square test for goodness of fit
    chi_square_stat = (n_overlap - expected_overlap) ** 2 / expected_overlap
    p_value = 1 - chi2.cdf(chi_square_stat, df=1)
    
    print(f"\nChi-square statistic: {chi_square_stat:.4f}")
    print(f"Degrees of freedom: 1")
    print(f"P-value: {p_value:.6f}")
    
    # Interpretation
    print(f"\n=== INTERPRETATION ===")
    if p_value < 0.001:
        print("*** HIGHLY SIGNIFICANT (p < 0.001) ***")
        significance = "highly significant"
    elif p_value < 0.01:
        print("** VERY SIGNIFICANT (p < 0.01) **")
        significance = "very significant"
    elif p_value < 0.05:
        print("* SIGNIFICANT (p < 0.05) *")
        significance = "significant"
    else:
        print("NOT SIGNIFICANT (p > 0.05)")
        significance = "not significant"
    
    if n_overlap < expected_overlap:
        print("→ Keywords are MORE DIVERGENT than expected by chance")
        print("→ Suggests systematic topic drift or specialization")
        interpretation = "divergent"
    else:
        print("→ Keywords are MORE CONVERGENT than expected by chance")
        print("→ Suggests topic stability across discussions")
        interpretation = "convergent"
    
    return {
        'chi_square': chi_square_stat,
        'p_value': p_value,
        'expected_overlap': expected_overlap,
        'observed_overlap': n_overlap,
        'significance': significance,
        'interpretation': interpretation
    }

def contingency_table_test(original_keywords, subtree_keywords, all_keywords):
    """
    Alternative approach using 2x2 contingency table.
    """
    
    print("\n=== 2x2 CONTINGENCY TABLE ANALYSIS ===")
    
    # Create contingency table
    overlap = original_keywords.intersection(subtree_keywords)
    original_only = original_keywords - subtree_keywords
    subtree_only = subtree_keywords - original_keywords
    neither = all_keywords - original_keywords - subtree_keywords
    
    # 2x2 table: [in_original, not_in_original] x [in_subtree, not_in_subtree]
    contingency_table = np.array([
        [len(overlap), len(subtree_only)],          # in_original: [both, subtree_only]
        [len(original_only), len(neither)]         # not_in_original: [original_only, neither]
    ])
    
    print("Contingency Table:")
    print("                In Subtree    Not in Subtree")
    print(f"In Original     {contingency_table[0,0]:8d}    {contingency_table[1,0]:12d}")
    print(f"Not in Original {contingency_table[0,1]:8d}    {contingency_table[1,1]:12d}")
    
    # Chi-square test
    chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)
    
    print(f"\nChi-square statistic: {chi2_stat:.4f}")
    print(f"P-value: {p_val:.6f}")
    print(f"Degrees of freedom: {dof}")
    
    # Effect size (Cramer's V)
    n = contingency_table.sum()
    cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
    
    print(f"Cramer's V (effect size): {cramers_v:.4f}")
    if cramers_v < 0.1:
        effect_size = "small"
    elif cramers_v < 0.3:
        effect_size = "medium"
    else:
        effect_size = "large"
    print(f"Effect size: {effect_size}")
    
    # Fisher's exact test (for small samples)
    if contingency_table.min() < 5:
        odds_ratio, fisher_p = fisher_exact(contingency_table)
        print(f"\nFisher's exact test p-value: {fisher_p:.6f}")
        print(f"Odds ratio: {odds_ratio:.4f}")
    
    return {
        'contingency_table': contingency_table,
        'chi2_stat': chi2_stat,
        'p_value': p_val,
        'cramers_v': cramers_v,
        'effect_size': effect_size
    }

def cluster_level_analysis(original_file, subtree_file):
    """
    Analyze convergence at the cluster level.
    """
    
    original_df = pd.read_csv(original_file)
    subtree_df = pd.read_csv(subtree_file)
    
    print("\n=== CLUSTER-LEVEL CONVERGENCE ANALYSIS ===")
    
    cluster_results = []
    
    for cluster_id in original_df['cluster_id'].unique():
        if cluster_id in subtree_df['cluster_id'].values:
            orig_cluster_keywords = set(original_df[original_df['cluster_id'] == cluster_id]['keyword'])
            sub_cluster_keywords = set(subtree_df[subtree_df['cluster_id'] == cluster_id]['keyword'])
            
            overlap = orig_cluster_keywords.intersection(sub_cluster_keywords)
            union = orig_cluster_keywords.union(sub_cluster_keywords)
            
            jaccard = len(overlap) / len(union) if len(union) > 0 else 0
            
            cluster_results.append({
                'cluster_id': cluster_id,
                'original_count': len(orig_cluster_keywords),
                'subtree_count': len(sub_cluster_keywords),
                'overlap_count': len(overlap),
                'jaccard_similarity': jaccard,
                'overlap_keywords': list(overlap)
            })
            
            print(f"Cluster {cluster_id}: {len(overlap)}/{len(orig_cluster_keywords)} overlap "
                  f"(Jaccard: {jaccard*100:.1f}%)")
    
    return pd.DataFrame(cluster_results)

def enhanced_dimension_analysis(original_file, subtree_file):
    """
    Enhanced analysis to identify which fractal dimension categories
    maintain topics best between original and subtree datasets.
    """
    
    original_df = pd.read_csv(original_file)
    subtree_df = pd.read_csv(subtree_file)
    
    print("\n=== ENHANCED FRACTAL DIMENSION CATEGORY ANALYSIS ===")
    
    # Get all unique dimension values and sort them
    all_dimensions = sorted(set(original_df['fract_dimension_type'].unique()) | 
                           set(subtree_df['fract_dimension_type'].unique()))
    
    dimension_results = []
    
    for dim_type in all_dimensions:
        orig_keywords = set(original_df[original_df['fract_dimension_type'] == dim_type]['keyword'])
        sub_keywords = set(subtree_df[subtree_df['fract_dimension_type'] == dim_type]['keyword'])
        
        if len(orig_keywords) > 0 and len(sub_keywords) > 0:
            overlap = orig_keywords.intersection(sub_keywords)
            union = orig_keywords.union(sub_keywords)
            
            # Multiple similarity metrics
            jaccard = len(overlap) / len(union) if len(union) > 0 else 0
            overlap_rate_orig = len(overlap) / len(orig_keywords)
            overlap_rate_sub = len(overlap) / len(sub_keywords)
            
            dimension_results.append({
                'dimension_type': dim_type,
                'original_count': len(orig_keywords),
                'subtree_count': len(sub_keywords),
                'overlap_count': len(overlap),
                'jaccard_similarity': jaccard,
                'overlap_rate_original': overlap_rate_orig,
                'overlap_rate_subtree': overlap_rate_sub,
                'mean_overlap_rate': (overlap_rate_orig + overlap_rate_sub) / 2,
                'overlapping_keywords': list(overlap)
            })
    
    # Convert to DataFrame and sort by topic maintenance
    results_df = pd.DataFrame(dimension_results)
    results_df = results_df.sort_values('jaccard_similarity', ascending=False)
    
    print("\n=== TOPIC MAINTENANCE RANKING (by Jaccard Similarity) ===")
    for idx, row in results_df.iterrows():
        print(f"{row['dimension_type']:>15}: {row['jaccard_similarity']*100:5.1f}% "
              f"({row['overlap_count']:2d}/{row['original_count']:2d} overlap)")
    
    # Identify highest and lowest performers
    best_category = results_df.iloc[0]
    worst_category = results_df.iloc[-1]
    
    print(f"\n=== KEY FINDINGS ===")
    print(f"🏆 BEST topic maintenance: {best_category['dimension_type']}")
    print(f"   - Jaccard similarity: {best_category['jaccard_similarity']*100:.1f}%")
    print(f"   - Overlap: {best_category['overlap_count']}/{best_category['original_count']} keywords")
    print(f"   - Overlapping topics: {', '.join(best_category['overlapping_keywords'][:5])}{'...' if len(best_category['overlapping_keywords']) > 5 else ''}")
    
    print(f"\n📉 WORST topic maintenance: {worst_category['dimension_type']}")
    print(f"   - Jaccard similarity: {worst_category['jaccard_similarity']*100:.1f}%")
    print(f"   - Overlap: {worst_category['overlap_count']}/{worst_category['original_count']} keywords")
    
    # Check if it's highest/lowest values
    dimension_values = [float(d) if d.replace('.','').isdigit() else d for d in all_dimensions if d.replace('.','').isdigit()]
    if dimension_values:
        highest_dim = max(dimension_values)
        lowest_dim = min(dimension_values)
        
        print(f"\n=== HIGHEST vs LOWEST DIMENSION VALUES ===")
        
        # Find results for highest and lowest values
        highest_result = results_df[results_df['dimension_type'] == str(highest_dim)]
        lowest_result = results_df[results_df['dimension_type'] == str(lowest_dim)]
        
        if not highest_result.empty:
            hr = highest_result.iloc[0]
            print(f"Highest dimension ({highest_dim}): {hr['jaccard_similarity']*100:.1f}% topic maintenance")
        
        if not lowest_result.empty:
            lr = lowest_result.iloc[0]
            print(f"Lowest dimension ({lowest_dim}): {lr['jaccard_similarity']*100:.1f}% topic maintenance")
    
    return results_df

def chi_square_highest_vs_lowest_dimensions(original_file, subtree_file):
    """
    Chi-square test comparing ONLY highest vs lowest dimension categories
    """
    original_df = pd.read_csv(original_file)
    subtree_df = pd.read_csv(subtree_file)
    
    print("\n=== CHI-SQUARE TEST: HIGHEST vs LOWEST DIMENSIONS ===")
    
    # Get all dimension categories from both datasets
    orig_dims = set(original_df['fract_dimension_type'].unique())
    sub_dims = set(subtree_df['fract_dimension_type'].unique())
    all_dims = orig_dims.union(sub_dims)
    
    print(f"Available dimension categories: {sorted(all_dims)}")
    
    # Check if we have both highest and lowest
    if 'highest' not in all_dims or 'lowest' not in all_dims:
        missing = []
        if 'highest' not in all_dims:
            missing.append('highest')
        if 'lowest' not in all_dims:
            missing.append('lowest')
        print(f"Error: Missing dimension categories: {missing}")
        return None
    
    print("Comparing: 'highest' vs 'lowest' dimensions")
    
    # Extract keywords for each category
    orig_high = set(original_df[original_df['fract_dimension_type'] == 'highest']['keyword'])
    orig_low = set(original_df[original_df['fract_dimension_type'] == 'lowest']['keyword'])
    sub_high = set(subtree_df[subtree_df['fract_dimension_type'] == 'highest']['keyword'])
    sub_low = set(subtree_df[subtree_df['fract_dimension_type'] == 'lowest']['keyword'])
    
    # Calculate overlaps
    high_overlap = len(orig_high.intersection(sub_high))
    high_total = len(orig_high)
    low_overlap = len(orig_low.intersection(sub_low))
    low_total = len(orig_low)
    
    print(f"\nHighest dimension:")
    print(f"  - Original keywords: {high_total}")
    print(f"  - Subtree keywords: {len(sub_high)}")
    print(f"  - Overlap: {high_overlap}")
    print(f"  - Overlap rate: {high_overlap/high_total*100:.1f}%" if high_total > 0 else "  - Overlap rate: N/A")
    
    print(f"\nLowest dimension:")
    print(f"  - Original keywords: {low_total}")
    print(f"  - Subtree keywords: {len(sub_low)}")
    print(f"  - Overlap: {low_overlap}")
    print(f"  - Overlap rate: {low_overlap/low_total*100:.1f}%" if low_total > 0 else "  - Overlap rate: N/A")
    
    # Check if we have enough data for chi-square test
    if high_total == 0 or low_total == 0:
        print("Error: One category has no keywords in original dataset")
        return None
    
    # Create contingency table: [overlap, no_overlap] for each dimension
    contingency = np.array([
        [high_overlap, high_total - high_overlap],    # highest dimension
        [low_overlap, low_total - low_overlap]        # lowest dimension
    ])
    
    print(f"\nContingency Table:")
    print(f"             Overlap    No Overlap")
    print(f"Highest:     {high_overlap:7d}    {high_total - high_overlap:10d}")
    print(f"Lowest:      {low_overlap:7d}    {low_total - low_overlap:10d}")
    
    # Chi-square test
    chi2_stat, p_val, dof, expected = chi2_contingency(contingency)
    
    print(f"\nChi-square Results:")
    print(f"Chi-square statistic: {chi2_stat:.4f}")
    print(f"P-value: {p_val:.6f}")
    print(f"Degrees of freedom: {dof}")
    
    # Effect size (Cramer's V)
    n = contingency.sum()
    cramers_v = np.sqrt(chi2_stat / (n * (min(contingency.shape) - 1)))
    print(f"Cramer's V (effect size): {cramers_v:.4f}")
    
    # Interpretation
    print(f"\n=== INTERPRETATION ===")
    if p_val < 0.05:
        print(f"*** SIGNIFICANT DIFFERENCE (p = {p_val:.4f}) ***")
        if high_overlap/high_total > low_overlap/low_total:
            print("→ HIGHEST dimension maintains topics better than LOWEST")
        else:
            print("→ LOWEST dimension maintains topics better than HIGHEST")
    else:
        print(f"No significant difference (p = {p_val:.4f})")
        print("→ Highest and lowest dimensions maintain topics equally well")
    
    return {
        'highest_dim': 'highest',
        'lowest_dim': 'lowest',
        'chi2_stat': chi2_stat,
        'p_value': p_val,
        'cramers_v': cramers_v,
        'high_overlap_rate': high_overlap/high_total if high_total > 0 else 0,
        'low_overlap_rate': low_overlap/low_total if low_total > 0 else 0
    }

def visualize_results(results_dict, overlap_keywords):
    """
    Create visualizations of the convergence analysis.
    """
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Observed vs Expected overlap
    categories = ['Expected\n(Random)', 'Observed\n(Actual)']
    values = [results_dict['expected_overlap'], results_dict['observed_overlap']]
    colors = ['lightcoral', 'steelblue']
    
    bars1 = ax1.bar(categories, values, color=colors, alpha=0.7)
    ax1.set_ylabel('Number of Overlapping Keywords')
    ax1.set_title('Expected vs Observed Keyword Overlap')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. P-value significance
    p_val = results_dict['p_value']
    significance_levels = [0.05, 0.01, 0.001]
    significance_labels = ['p < 0.05', 'p < 0.01', 'p < 0.001']
    colors_sig = ['yellow', 'orange', 'red']
    
    ax2.barh(range(len(significance_levels)), significance_levels, color=colors_sig, alpha=0.5)
    ax2.axvline(x=p_val, color='blue', linestyle='--', linewidth=3, label=f'Actual p-value: {p_val:.4f}')
    ax2.set_yticks(range(len(significance_levels)))
    ax2.set_yticklabels(significance_labels)
    ax2.set_xlabel('P-value')
    ax2.set_title('Statistical Significance Test')
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)
    
    # 3. Keyword overlap heatmap (simplified)
    overlap_data = np.array([[len(overlap_keywords), 60-len(overlap_keywords)],
                            [60-len(overlap_keywords), 107-60-60+len(overlap_keywords)]])
    
    sns.heatmap(overlap_data, annot=True, fmt='d', cmap='Blues',
                xticklabels=['In Subtree', 'Not in Subtree'],
                yticklabels=['In Original', 'Not in Original'], ax=ax3)
    ax3.set_title('Keyword Distribution Heatmap')
    
    # 4. Chi-square interpretation
    chi_sq = results_dict['chi_square']
    critical_values = [3.841, 6.635, 10.828]  # for df=1
    critical_labels = ['p=0.05', 'p=0.01', 'p=0.001']
    
    ax4.barh(range(len(critical_values)), critical_values, color='lightgray', alpha=0.7)
    ax4.axvline(x=chi_sq, color='red', linestyle='-', linewidth=3, 
               label=f'Chi-square: {chi_sq:.3f}')
    ax4.set_yticks(range(len(critical_values)))
    ax4.set_yticklabels(critical_labels)
    ax4.set_xlabel('Chi-square Value')
    ax4.set_title('Chi-square Critical Values')
    ax4.legend()
    ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Main analysis function
def main():
    """
    Run complete keyword convergence analysis.
    """
    
    # File paths - update these to your actual file paths
    original_file = 'clustering_results.csv'
    subtree_file = 'subtree_clustering_results.csv'
    
    try:
        # Basic convergence analysis
        original_kw, subtree_kw, overlap, all_kw = analyze_keyword_convergence(original_file, subtree_file)
        
        # Chi-square tests
        independence_results = chi_square_independence_test(original_kw, subtree_kw, all_kw)
        contingency_results = contingency_table_test(original_kw, subtree_kw, all_kw)
        
        # Detailed analyses
        cluster_df = cluster_level_analysis(original_file, subtree_file)
        
        # Enhanced dimension analysis
        dimension_results = enhanced_dimension_analysis(original_file, subtree_file)
        
        # Chi-square test comparing highest vs lowest dimensions
        extreme_comparison = chi_square_highest_vs_lowest_dimensions(original_file, subtree_file)
        
        # Print overlapping keywords
        print(f"\n=== OVERLAPPING KEYWORDS ({len(overlap)}) ===")
        for i, keyword in enumerate(sorted(overlap), 1):
            print(f"{i:2d}. {keyword}")
        
        # Visualizations
        visualize_results(independence_results, overlap)
        
        # Summary
        print(f"\n=== RESEARCH IMPLICATIONS ===")
        print(f"• Chi-square test shows {independence_results['significance']} difference from random")
        print(f"• Keywords are {independence_results['interpretation']} between original and subtree")
        print(f"• Effect size: {contingency_results['effect_size']} (Cramer's V = {contingency_results['cramers_v']:.3f})")
        
        if independence_results['interpretation'] == 'divergent':
            print("• This suggests systematic topic evolution in discussion threads")
            print("• Consider investigating what drives this semantic drift")
        else:
            print("• This suggests strong topic stability across discussion levels")
            print("• Consider investigating what maintains this semantic coherence")
        
        return {
            'independence_test': independence_results,
            'contingency_test': contingency_results,
            'cluster_analysis': cluster_df,
            'dimension_results': dimension_results,
            'extreme_comparison': extreme_comparison,
            'overlapping_keywords': list(overlap)
        }
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file {e.filename}")
        print("Please update the file paths in the main() function")
        return None
    except Exception as e:
        print(f"Error during analysis: {e}")
        return None

if __name__ == "__main__":
    results = main()