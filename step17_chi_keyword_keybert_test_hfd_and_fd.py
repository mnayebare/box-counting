import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, chi2
from scipy.stats import fisher_exact
import matplotlib.pyplot as plt
import seaborn as sns

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

def dimension_type_analysis(original_file, subtree_file):
    """
    Analyze convergence by fractal dimension type.
    """
    
    original_df = pd.read_csv(original_file)
    subtree_df = pd.read_csv(subtree_file)
    
    print("\n=== FRACTAL DIMENSION TYPE ANALYSIS ===")
    
    dimension_results = []
    
    for dim_type in original_df['fract_dimension_type'].unique():
        if dim_type in subtree_df['fract_dimension_type'].values:
            orig_keywords = set(original_df[original_df['fract_dimension_type'] == dim_type]['keyword'])
            sub_keywords = set(subtree_df[subtree_df['fract_dimension_type'] == dim_type]['keyword'])
            
            overlap = orig_keywords.intersection(sub_keywords)
            union = orig_keywords.union(sub_keywords)
            
            jaccard = len(overlap) / len(union) if len(union) > 0 else 0
            
            dimension_results.append({
                'dimension_type': dim_type,
                'original_count': len(orig_keywords),
                'subtree_count': len(sub_keywords),
                'overlap_count': len(overlap),
                'jaccard_similarity': jaccard
            })
            
            print(f"{dim_type}: {len(overlap)}/{len(orig_keywords)} overlap "
                  f"(Jaccard: {jaccard*100:.1f}%)")
    
    return pd.DataFrame(dimension_results)

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
        dimension_df = dimension_type_analysis(original_file, subtree_file)
        
        # Print overlapping keywords
        print(f"\n=== OVERLAPPING KEYWORDS ({len(overlap)}) ===")
        for i, keyword in enumerate(sorted(overlap), 1):
            print(f"{i:2d}. {keyword}")
    
        
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
            'dimension_analysis': dimension_df,
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