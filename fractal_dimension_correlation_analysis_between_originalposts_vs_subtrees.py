import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import numpy as np

# Set publication-quality style with black and white theme
plt.style.use('default')

# Configure matplotlib for publication quality black and white
plt.rcParams.update({
    'font.size': 14,
    'font.weight': 'bold',
    'axes.labelsize': 16,
    'axes.labelweight': 'bold',
    'axes.titlesize': 18,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 20,
    'figure.titleweight': 'bold',
    'axes.linewidth': 1.5,
    'grid.linewidth': 0.8,
    'grid.alpha': 0.3,
    'axes.edgecolor': 'black',
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white'
})

# Load the datasets
original_posts_df = pd.read_csv('original_posts_fractal_analysis_results.csv')
subtree_df = pd.read_csv('subtree_fractal_analysis.csv')

# Clean column names (remove any whitespace)
original_posts_df.columns = original_posts_df.columns.str.strip()
subtree_df.columns = subtree_df.columns.str.strip()

# Function to merge and correlate data for a specific conversation type
def analyze_correlation(conversation_type_name, original_df, subtree_df):
    # Filter data for the specified conversation type
    original_filtered = original_df[original_df['conversation_type'] == conversation_type_name].copy()
    subtree_filtered = subtree_df[subtree_df['conversation_type'] == conversation_type_name].copy()
    
    # Merge on post_id to get matching pairs
    merged_data = pd.merge(
        original_filtered[['post_id', 'fractal_dimension']], 
        subtree_filtered[['post_id', 'fractal_dimension']], 
        on='post_id', 
        suffixes=('_original', '_subtree')
    )
    
    return merged_data

# Analyze correlations for both conversation types
richly_branching_data = analyze_correlation('richly branching', original_posts_df, subtree_df)
poorly_branching_data = analyze_correlation('poorly branching', original_posts_df, subtree_df)

# Create the correlation plots with black and white styling
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Define black and white colors and patterns
rich_color = 'black'           # Black for richly branching
poor_color = 'white'           # White with black edge for poorly branching
line_color = 'black'           # Black for trend lines
grid_color = '#666666'         # Dark gray for grid

# Graph 1: Richly Branching Conversations
if not richly_branching_data.empty:
    # Create scatter plot with black filled circles
    scatter1 = ax1.scatter(richly_branching_data['fractal_dimension_original'], 
                          richly_branching_data['fractal_dimension_subtree'], 
                          alpha=0.8, color=rich_color, s=80, 
                          edgecolors='black', linewidth=1.0, 
                          marker='o', label='Richly Branching')
    
    # Calculate correlation
    pearson_r_rich, p_value_rich = pearsonr(richly_branching_data['fractal_dimension_original'], 
                                           richly_branching_data['fractal_dimension_subtree'])
    
    # Add trend line
    z = np.polyfit(richly_branching_data['fractal_dimension_original'], 
                   richly_branching_data['fractal_dimension_subtree'], 1)
    p = np.poly1d(z)
    ax1.plot(richly_branching_data['fractal_dimension_original'], 
             p(richly_branching_data['fractal_dimension_original']), 
             color=line_color, linestyle='-', alpha=1.0, linewidth=2)
    
    # Enhanced labels and formatting
    ax1.set_xlabel('Original Post Fractal Dimension', fontweight='bold', fontsize=16)
    ax1.set_ylabel('Subtree Fractal Dimension', fontweight='bold', fontsize=16)
    
    # Enhanced title with statistical information
    title_text = f'Richly Branching Conversations\nr = {pearson_r_rich:.3f}, p = {p_value_rich:.3f}, n = {len(richly_branching_data)}'
    ax1.set_title(title_text, fontweight='bold', fontsize=16, pad=20)
    
    # Enhanced grid with gray color
    ax1.grid(True, alpha=0.4, linewidth=0.8, linestyle='--', color=grid_color)
    ax1.set_facecolor('white')
    
    # Add black frame
    for spine in ax1.spines.values():
        spine.set_linewidth(2.0)
        spine.set_color('black')
    
else:
    ax1.text(0.5, 0.5, 'No richly branching data found', 
             ha='center', va='center', transform=ax1.transAxes, 
             fontweight='bold', fontsize=14, color='black')
    ax1.set_title('Richly Branching Conversations - No Data', fontweight='bold')

# Graph 2: Poorly Branching Conversations
if not poorly_branching_data.empty:
    # Create scatter plot with white filled circles and black edges
    scatter2 = ax2.scatter(poorly_branching_data['fractal_dimension_original'], 
                          poorly_branching_data['fractal_dimension_subtree'], 
                          alpha=1.0, facecolors=poor_color, s=80, 
                          edgecolors='black', linewidth=1.5, 
                          marker='o', label='Poorly Branching')  # Round markers
    
    # Calculate correlation
    pearson_r_poor, p_value_poor = pearsonr(poorly_branching_data['fractal_dimension_original'], 
                                           poorly_branching_data['fractal_dimension_subtree'])
    
    # Add trend line with dashed style for distinction
    z = np.polyfit(poorly_branching_data['fractal_dimension_original'], 
                   poorly_branching_data['fractal_dimension_subtree'], 1)
    p = np.poly1d(z)
    ax2.plot(poorly_branching_data['fractal_dimension_original'], 
             p(poorly_branching_data['fractal_dimension_original']), 
             color=line_color, linestyle='--', alpha=1.0, linewidth=2)
    
    # Enhanced labels and formatting
    ax2.set_xlabel('Original Post Fractal Dimension', fontweight='bold', fontsize=16)
    ax2.set_ylabel('Subtree Fractal Dimension', fontweight='bold', fontsize=16)
    
    # Enhanced title with statistical information
    title_text = f'Poorly Branching Conversations\nr = {pearson_r_poor:.3f}, p = {p_value_poor:.3f}, n = {len(poorly_branching_data)}'
    ax2.set_title(title_text, fontweight='bold', fontsize=16, pad=20)
    
    # Enhanced grid with gray color
    ax2.grid(True, alpha=0.4, linewidth=0.8, linestyle='--', color=grid_color)
    ax2.set_facecolor('white')
    
    # Add black frame
    for spine in ax2.spines.values():
        spine.set_linewidth(2.0)
        spine.set_color('black')
    
else:
    ax2.text(0.5, 0.5, 'No poorly branching data found', 
             ha='center', va='center', transform=ax2.transAxes,
             fontweight='bold', fontsize=14, color='black')
    ax2.set_title('Poorly Branching Conversations - No Data', fontweight='bold')

# Add overall figure title
fig.suptitle('Fractal Dimension Correlation Analysis: Original Posts vs. Subtrees', 
             fontsize=20, fontweight='bold', y=0.98, color='black')

# Adjust layout with better spacing
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save the correlation plots with high quality settings in black and white
plt.savefig('fractal_dimension_correlation_analysis_bw.png', 
            dpi=600, bbox_inches='tight', 
            facecolor='white', edgecolor='black',
            format='png')
plt.savefig('fractal_dimension_correlation_analysis_bw.pdf', 
            bbox_inches='tight', 
            facecolor='white', edgecolor='black',
            format='pdf')
plt.savefig('fractal_dimension_correlation_analysis_bw.svg', 
            bbox_inches='tight', 
            facecolor='white', edgecolor='black',
            format='svg')

print("Enhanced BLACK AND WHITE correlation plots saved as:")
print("- fractal_dimension_correlation_analysis_bw.png (600 DPI, publication quality)")
print("- fractal_dimension_correlation_analysis_bw.pdf (vector format)")
print("- fractal_dimension_correlation_analysis_bw.svg (vector format, ideal for journals)")

plt.show()

# Print detailed statistics with enhanced formatting
print("="*80)
print("FRACTAL DIMENSION CORRELATION ANALYSIS - RESEARCH SUMMARY")
print("="*80)

print("\n1. RICHLY BRANCHING CONVERSATIONS:")
if not richly_branching_data.empty:
    print(f"   Sample size (n): {len(richly_branching_data)}")
    print(f"   Pearson correlation coefficient (r): {pearson_r_rich:.4f}")
    print(f"   Statistical significance (p-value): {p_value_rich:.4f}")
    
    # Determine significance level
    if p_value_rich < 0.001:
        sig_level = "***"
    elif p_value_rich < 0.01:
        sig_level = "**"
    elif p_value_rich < 0.05:
        sig_level = "*"
    else:
        sig_level = "ns"
    print(f"   Significance level: {sig_level}")
    
    spearman_r_rich, spearman_p_rich = spearmanr(richly_branching_data['fractal_dimension_original'], 
                                                 richly_branching_data['fractal_dimension_subtree'])
    print(f"   Spearman rank correlation (ρ): {spearman_r_rich:.4f} (p = {spearman_p_rich:.4f})")
    print(f"   Original posts - Mean ± SD: {richly_branching_data['fractal_dimension_original'].mean():.4f} ± {richly_branching_data['fractal_dimension_original'].std():.4f}")
    print(f"   Subtrees - Mean ± SD: {richly_branching_data['fractal_dimension_subtree'].mean():.4f} ± {richly_branching_data['fractal_dimension_subtree'].std():.4f}")
    
    # Effect size interpretation
    if abs(pearson_r_rich) >= 0.7:
        effect_size = "large"
    elif abs(pearson_r_rich) >= 0.3:
        effect_size = "medium"
    elif abs(pearson_r_rich) >= 0.1:
        effect_size = "small"
    else:
        effect_size = "negligible"
    print(f"   Effect size interpretation: {effect_size}")
else:
    print("   No data available")

print("\n2. POORLY BRANCHING CONVERSATIONS:")
if not poorly_branching_data.empty:
    print(f"   Sample size (n): {len(poorly_branching_data)}")
    print(f"   Pearson correlation coefficient (r): {pearson_r_poor:.4f}")
    print(f"   Statistical significance (p-value): {p_value_poor:.4f}")
    
    # Determine significance level
    if p_value_poor < 0.001:
        sig_level = "***"
    elif p_value_poor < 0.01:
        sig_level = "**"
    elif p_value_poor < 0.05:
        sig_level = "*"
    else:
        sig_level = "ns"
    print(f"   Significance level: {sig_level}")
    
    spearman_r_poor, spearman_p_poor = spearmanr(poorly_branching_data['fractal_dimension_original'], 
                                                 poorly_branching_data['fractal_dimension_subtree'])
    print(f"   Spearman rank correlation (ρ): {spearman_r_poor:.4f} (p = {spearman_p_poor:.4f})")
    print(f"   Original posts - Mean ± SD: {poorly_branching_data['fractal_dimension_original'].mean():.4f} ± {poorly_branching_data['fractal_dimension_original'].std():.4f}")
    print(f"   Subtrees - Mean ± SD: {poorly_branching_data['fractal_dimension_subtree'].mean():.4f} ± {poorly_branching_data['fractal_dimension_subtree'].std():.4f}")
    
    # Effect size interpretation
    if abs(pearson_r_poor) >= 0.7:
        effect_size = "large"
    elif abs(pearson_r_poor) >= 0.3:
        effect_size = "medium"
    elif abs(pearson_r_poor) >= 0.1:
        effect_size = "small"
    else:
        effect_size = "negligible"
    print(f"   Effect size interpretation: {effect_size}")
else:
    print("   No data available")

# Additional analysis: Check available conversation types
print("\n3. DATA OVERVIEW:")
print("   Available conversation types in original posts:")
print(f"   {original_posts_df['conversation_type'].value_counts().to_dict()}")
print("   Available conversation types in subtrees:")
print(f"   {subtree_df['conversation_type'].value_counts().to_dict()}")

# Create a summary dataframe for easy reference with enhanced formatting
if not richly_branching_data.empty and not poorly_branching_data.empty:
    summary_data = {
        'Conversation_Type': ['Richly Branching', 'Poorly Branching'],
        'Sample_Size': [len(richly_branching_data), len(poorly_branching_data)],
        'Pearson_r': [pearson_r_rich, pearson_r_poor],
        'Pearson_p_value': [p_value_rich, p_value_poor],
        'Original_Mean_FD': [richly_branching_data['fractal_dimension_original'].mean(), 
                            poorly_branching_data['fractal_dimension_original'].mean()],
        'Original_SD_FD': [richly_branching_data['fractal_dimension_original'].std(), 
                          poorly_branching_data['fractal_dimension_original'].std()],
        'Subtree_Mean_FD': [richly_branching_data['fractal_dimension_subtree'].mean(), 
                           poorly_branching_data['fractal_dimension_subtree'].mean()],
        'Subtree_SD_FD': [richly_branching_data['fractal_dimension_subtree'].std(), 
                         poorly_branching_data['fractal_dimension_subtree'].std()]
    }
    
    summary_df = pd.DataFrame(summary_data)
    print("\n4. RESEARCH SUMMARY TABLE:")
    print("="*80)
    print(summary_df.round(4).to_string(index=False))
    
    # Save summary table for research paper
    summary_df.round(4).to_csv('fractal_correlation_summary_table_bw.csv', index=False)
    print("\nSummary table saved as: fractal_correlation_summary_table_bw.csv")

print("\n" + "="*80)
print("BLACK AND WHITE ANALYSIS COMPLETE - Files ready for research publication")
print("="*80)