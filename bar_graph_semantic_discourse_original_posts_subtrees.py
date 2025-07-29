import pandas as pd
import numpy as np
from tabulate import tabulate

def load_and_prepare_data():
    """
    Load data and prepare for manuscript table
    """
    # Load the summary files
    original_posts = pd.read_csv('original_posts_discourse_markers_summary.csv')
    subtrees = pd.read_csv('deepest_subtrees_summary.csv')
    
    marker_columns = [
        'comments_with_disagreement_markers',
        'comments_with_agreement_markers', 
        'comments_with_sarcasm_markers',
        'comments_with_weak_markers'
    ]
    
    return original_posts, subtrees, marker_columns

def create_manuscript_table():
    """
    Create a comprehensive table suitable for research manuscript
    """
    original_posts, subtrees, marker_columns = load_and_prepare_data()
    
    # Calculate statistics for original posts
    original_stats = original_posts.groupby('conversation_type')[marker_columns].agg(['mean', 'std', 'count']).round(2)
    
    # Calculate statistics for subtrees  
    subtree_stats = subtrees.groupby('conversation_type')[marker_columns].agg(['mean', 'std', 'count']).round(2)
    
    # Calculate total comments/replies analyzed for each group
    orig_richly_total_comments = original_posts[original_posts['conversation_type'] == 'richly branching']['total_comments'].sum()
    orig_poorly_total_comments = original_posts[original_posts['conversation_type'] == 'poorly branching']['total_comments'].sum()
    sub_richly_total_comments = subtrees[subtrees['conversation_type'] == 'richly branching']['total_comments'].sum()
    sub_poorly_total_comments = subtrees[subtrees['conversation_type'] == 'poorly branching']['total_comments'].sum()
    
    # Create manuscript table data
    table_data = []
    
    # Headers
    marker_names = ['Disagreement', 'Agreement', 'Sarcasm', 'Weak']
    
    for i, (marker, marker_name) in enumerate(zip(marker_columns, marker_names)):
        # Original Posts - Richly Branching
        orig_richly_mean = original_stats.loc['richly branching', (marker, 'mean')]
        orig_richly_std = original_stats.loc['richly branching', (marker, 'std')]
        
        # Original Posts - Poorly Branching  
        orig_poorly_mean = original_stats.loc['poorly branching', (marker, 'mean')]
        orig_poorly_std = original_stats.loc['poorly branching', (marker, 'std')]
        
        # Subtrees - Richly Branching
        sub_richly_mean = subtree_stats.loc['richly branching', (marker, 'mean')]
        sub_richly_std = subtree_stats.loc['richly branching', (marker, 'std')]
        
        # Subtrees - Poorly Branching
        sub_poorly_mean = subtree_stats.loc['poorly branching', (marker, 'mean')]
        sub_poorly_std = subtree_stats.loc['poorly branching', (marker, 'std')]
        
        table_data.append([
            marker_name,
            f"{orig_richly_mean:.1f} ± {orig_richly_std:.1f}",
            f"{orig_poorly_mean:.1f} ± {orig_poorly_std:.1f}",
            f"{sub_richly_mean:.1f} ± {sub_richly_std:.1f}",
            f"{sub_poorly_mean:.1f} ± {sub_poorly_std:.1f}",
            f"{orig_richly_total_comments:,}/{orig_poorly_total_comments:,}",  # Total replies analyzed
            f"{sub_richly_total_comments:,}/{sub_poorly_total_comments:,}"     # Total replies analyzed
        ])
    
    return table_data

def create_detailed_manuscript_table():
    """
    Create detailed table with additional statistics and effect sizes
    """
    original_posts, subtrees, marker_columns = load_and_prepare_data()
    
    # Calculate comprehensive statistics
    def calculate_comprehensive_stats(df, group_col, marker_cols):
        stats = df.groupby(group_col)[marker_cols].agg([
            'mean', 'std', 'median', 'count', 'min', 'max'
        ]).round(2)
        return stats
    
    original_stats = calculate_comprehensive_stats(original_posts, 'conversation_type', marker_columns)
    subtree_stats = calculate_comprehensive_stats(subtrees, 'conversation_type', marker_columns)
    
    # Calculate total comments/replies analyzed for each group
    orig_richly_total_comments = original_posts[original_posts['conversation_type'] == 'richly branching']['total_comments'].sum()
    orig_poorly_total_comments = original_posts[original_posts['conversation_type'] == 'poorly branching']['total_comments'].sum()
    sub_richly_total_comments = subtrees[subtrees['conversation_type'] == 'richly branching']['total_comments'].sum()
    sub_poorly_total_comments = subtrees[subtrees['conversation_type'] == 'poorly branching']['total_comments'].sum()
    
    # Create detailed table
    detailed_data = []
    marker_names = ['Disagreement', 'Agreement', 'Sarcasm', 'Weak']
    
    for marker, marker_name in zip(marker_columns, marker_names):
        # Get all statistics
        orig_richly = original_stats.loc['richly branching', marker]
        orig_poorly = original_stats.loc['poorly branching', marker]
        sub_richly = subtree_stats.loc['richly branching', marker]
        sub_poorly = subtree_stats.loc['poorly branching', marker]
        
        # Calculate effect sizes (Cohen's d approximation)
        def cohens_d(mean1, std1, mean2, std2):
            pooled_std = np.sqrt(((std1 ** 2) + (std2 ** 2)) / 2)
            return (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        
        # Effect size: Richly vs Poorly in Original Posts
        orig_effect = cohens_d(orig_richly['mean'], orig_richly['std'], 
                              orig_poorly['mean'], orig_poorly['std'])
        
        # Effect size: Richly vs Poorly in Subtrees
        sub_effect = cohens_d(sub_richly['mean'], sub_richly['std'],
                             sub_poorly['mean'], sub_poorly['std'])
        
        detailed_data.append([
            marker_name,
            f"{orig_richly['mean']:.1f}",
            f"{orig_richly['std']:.1f}",
            f"{orig_poorly['mean']:.1f}",
            f"{orig_poorly['std']:.1f}",
            f"{sub_richly['mean']:.1f}",
            f"{sub_richly['std']:.1f}",
            f"{sub_poorly['mean']:.1f}",
            f"{sub_poorly['std']:.1f}",
            f"{orig_effect:.2f}",
            f"{sub_effect:.2f}",
            f"{orig_richly_total_comments:,}/{orig_poorly_total_comments:,}",  # Total replies analyzed
            f"{sub_richly_total_comments:,}/{sub_poorly_total_comments:,}"     # Total replies analyzed
        ])
    
    return detailed_data

def print_manuscript_tables():
    """
    Print formatted tables suitable for research manuscript
    """
    print("=" * 80)
    print("RESEARCH MANUSCRIPT TABLES")
    print("=" * 80)
    
    # Table 1: Basic comparison
    print("\nTABLE 1: Discourse Marker Frequencies by Conversation Type and Analysis Method")
    print("-" * 80)
    
    table_data = create_manuscript_table()
    
    headers = [
        "Discourse Marker",
        "Original Posts\nRichly Branching\nM ± SD",
        "Original Posts\nPoorly Branching\nM ± SD", 
        "Subtrees\nRichly Branching\nM ± SD",
        "Subtrees\nPoorly Branching\nM ± SD",
        "N (Original)\nRich/Poor",  # Clear header for separate N values
        "N (Subtrees)\nRich/Poor"   # Clear header for separate N values
    ]
    
    print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="center"))
    
    # Table 2: Detailed statistics
    print("\n\nTABLE 2: Detailed Statistical Comparison of Discourse Markers")
    print("-" * 120)
    
    detailed_data = create_detailed_manuscript_table()
    
    detailed_headers = [
        "Marker",
        "Orig-Rich\nMean",
        "Orig-Rich\nSD",
        "Orig-Poor\nMean", 
        "Orig-Poor\nSD",
        "Sub-Rich\nMean",
        "Sub-Rich\nSD",
        "Sub-Poor\nMean",
        "Sub-Poor\nSD",
        "Effect Size\n(Orig)",
        "Effect Size\n(Sub)",
        "N-Orig\nRich/Poor",  # Clear header for separate N values
        "N-Sub\nRich/Poor"    # Clear header for separate N values
    ]
    
    print(tabulate(detailed_data, headers=detailed_headers, tablefmt="grid", stralign="center"))

def create_latex_table():
    """
    Generate LaTeX table code for manuscript
    """
    print("\n\nLATEX TABLE CODE:")
    print("-" * 50)
    
    table_data = create_manuscript_table()
    
    latex_code = """
\\begin{table}[htbp]
\\centering
\\caption{Discourse Marker Frequencies by Conversation Type and Analysis Method}
\\label{tab:discourse_markers}
\\begin{tabular}{lcccccc}
\\hline
\\textbf{Discourse Marker} & \\multicolumn{2}{c}{\\textbf{Original Posts}} & \\multicolumn{2}{c}{\\textbf{Subtrees}} & \\multicolumn{2}{c}{\\textbf{Sample Size}} \\\\
\\cline{2-3} \\cline{4-5} \\cline{6-7}
& Richly & Poorly & Richly & Poorly & Original & Subtrees \\\\
& Branching & Branching & Branching & Branching & Posts & \\\\
& M ± SD & M ± SD & M ± SD & M ± SD & Rich/Poor & Rich/Poor \\\\
\\hline
"""
    
    for row in table_data:
        latex_code += f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} & {row[5]} & {row[6]} \\\\\n"
    
    latex_code += """\\hline
\\end{tabular}
\\begin{tablenotes}
\\small
\\item Note: M = Mean, SD = Standard Deviation. Sample sizes shown as Richly Branching/Poorly Branching. Richly branching conversations had more than X responses, poorly branching had fewer than X responses.
\\end{tablenotes}
\\end{table}
"""
    
    print(latex_code)

def create_summary_statistics():
    """
    Create summary statistics for manuscript text
    """
    original_posts, subtrees, marker_columns = load_and_prepare_data()
    
    print("\n\nSUMMARY STATISTICS FOR MANUSCRIPT TEXT:")
    print("-" * 50)
    
    # Overall dataset statistics
    total_orig = len(original_posts)
    richly_orig = len(original_posts[original_posts['conversation_type'] == 'richly branching'])
    poorly_orig = len(original_posts[original_posts['conversation_type'] == 'poorly branching'])
    
    total_sub = len(subtrees)
    richly_sub = len(subtrees[subtrees['conversation_type'] == 'richly branching'])
    poorly_sub = len(subtrees[subtrees['conversation_type'] == 'poorly branching'])
    
    print(f"Dataset Overview:")
    print(f"- Original Posts: {total_orig} conversations")
    print(f"  - Richly branching: {richly_orig} ({richly_orig/total_orig*100:.1f}%)")
    print(f"  - Poorly branching: {poorly_orig} ({poorly_orig/total_orig*100:.1f}%)")
    
    print(f"- Subtrees: {total_sub} subtrees")
    print(f"  - From richly branching: {richly_sub} ({richly_sub/total_sub*100:.1f}%)")
    print(f"  - From poorly branching: {poorly_sub} ({poorly_sub/total_sub*100:.1f}%)")
    
    # Key findings
    marker_names = ['Disagreement', 'Agreement', 'Sarcasm', 'Weak']
    
    print(f"\nKey Findings:")
    for marker, marker_name in zip(marker_columns, marker_names):
        orig_richly = original_posts[original_posts['conversation_type'] == 'richly branching'][marker].mean()
        orig_poorly = original_posts[original_posts['conversation_type'] == 'poorly branching'][marker].mean()
        sub_richly = subtrees[subtrees['conversation_type'] == 'richly branching'][marker].mean()
        sub_poorly = subtrees[subtrees['conversation_type'] == 'poorly branching'][marker].mean()
        
        print(f"\n{marker_name} Markers:")
        print(f"  - Original Posts: Richly ({orig_richly:.1f}) vs Poorly ({orig_poorly:.1f}) branching")
        print(f"  - Subtrees: Richly ({sub_richly:.1f}) vs Poorly ({sub_poorly:.1f}) branching")
        
        # Direction of differences
        orig_direction = "higher" if orig_richly > orig_poorly else "lower"
        sub_direction = "higher" if sub_richly > sub_poorly else "lower"
        print(f"  - Richly branching shows {orig_direction} {marker_name.lower()} in original posts")
        print(f"  - Richly branching shows {sub_direction} {marker_name.lower()} in subtrees")

def main():
    """
    Generate all manuscript tables and statistics
    """
    # Check if tabulate is available
    try:
        from tabulate import tabulate
    except ImportError:
        print("Installing tabulate for better table formatting...")
        import subprocess
        subprocess.check_call(["pip", "install", "tabulate"])
        from tabulate import tabulate
    
    print_manuscript_tables()
    create_latex_table()
    create_summary_statistics()

if __name__ == "__main__":
    main()