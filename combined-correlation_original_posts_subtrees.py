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
original_posts_df = pd.read_csv('original_posts_fractal_analysis.csv')
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
richly_branching_data = analyze_correlation('controversial', original_posts_df, subtree_df)
poorly_branching_data = analyze_correlation('technical', original_posts_df, subtree_df)

# Create single combined correlation plot
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Define colors and styles
rich_color = 'black'
poor_color = 'white'
line_color = 'black'
grid_color = '#666666'

# Initialize correlation variables
pearson_r_rich = pearson_r_poor = p_value_rich = p_value_poor = None

# Plot Controversial conversations (black filled circles)
if not richly_branching_data.empty:
    scatter1 = ax.scatter(richly_branching_data['fractal_dimension_original'], 
                         richly_branching_data['fractal_dimension_subtree'], 
                         alpha=0.8, color=rich_color, s=80, 
                         edgecolors='black', linewidth=1.0, 
                         marker='o', label=f'Controversial (n={len(richly_branching_data)})')
    
    # Calculate correlation for controversial
    pearson_r_rich, p_value_rich = pearsonr(richly_branching_data['fractal_dimension_original'], 
                                           richly_branching_data['fractal_dimension_subtree'])
    
    # Add trend line for controversial
    if len(richly_branching_data) > 1:
        z1 = np.polyfit(richly_branching_data['fractal_dimension_original'], 
                       richly_branching_data['fractal_dimension_subtree'], 1)
        p1 = np.poly1d(z1)
        ax.plot(richly_branching_data['fractal_dimension_original'], 
                p1(richly_branching_data['fractal_dimension_original']), 
                color=line_color, linestyle='-', alpha=0.8, linewidth=2)

# Plot Technical conversations (white filled circles with black edges)
if not poorly_branching_data.empty:
    scatter2 = ax.scatter(poorly_branching_data['fractal_dimension_original'], 
                         poorly_branching_data['fractal_dimension_subtree'], 
                         alpha=1.0, facecolors=poor_color, s=80, 
                         edgecolors='black', linewidth=1.5, 
                         marker='o', label=f'Technical (n={len(poorly_branching_data)})')
    
    # Calculate correlation for technical
    pearson_r_poor, p_value_poor = pearsonr(poorly_branching_data['fractal_dimension_original'], 
                                           poorly_branching_data['fractal_dimension_subtree'])
    
    # Add trend line for technical (dashed)
    if len(poorly_branching_data) > 1:
        z2 = np.polyfit(poorly_branching_data['fractal_dimension_original'], 
                       poorly_branching_data['fractal_dimension_subtree'], 1)
        p2 = np.poly1d(z2)
        ax.plot(poorly_branching_data['fractal_dimension_original'], 
                p2(poorly_branching_data['fractal_dimension_original']), 
                color=line_color, linestyle='--', alpha=0.8, linewidth=2)

# Enhanced labels and formatting
ax.set_xlabel('Original Posts', fontweight='bold', fontsize=16)
ax.set_ylabel('Sub-trees', fontweight='bold', fontsize=16)

# Combined title with both correlation statistics
title_parts = []
if not richly_branching_data.empty and pearson_r_rich is not None:
    title_parts.append(f'Controversial: r = {pearson_r_rich:.3f}, p = {p_value_rich:.3f}')
if not poorly_branching_data.empty and pearson_r_poor is not None:
    title_parts.append(f'Technical: r = {pearson_r_poor:.3f}, p = {p_value_poor:.3f}')

if title_parts:
    title_text = 'Fractal Dimension Correlations\n' + ' | '.join(title_parts)
else:
    title_text = 'Fractal Dimension Correlations - No Data Available'

ax.set_title(title_text, fontweight='bold', fontsize=16, pad=20)

# Enhanced grid and styling
ax.grid(True, alpha=0.4, linewidth=0.8, linestyle='--', color=grid_color)
ax.set_facecolor('white')

# Add black frame
for spine in ax.spines.values():
    spine.set_linewidth(2.0)
    spine.set_color('black')

# Create the legend first, then set fontweight on the text objects
legend = ax.legend(loc='best', frameon=True, fancybox=False, shadow=False, 
                  framealpha=1.0, edgecolor='black', facecolor='white',
                  fontsize=12)
# Set legend text to bold
for text in legend.get_texts():
    text.set_fontweight('bold')

# Adjust layout
plt.tight_layout()

# Save the combined plot with high quality settings
plt.savefig('combined_fractal_correlation_bw.png', 
            dpi=600, bbox_inches='tight', 
            facecolor='white', edgecolor='black',
            format='png')

print("Enhanced BLACK AND WHITE combined correlation plot saved as:")
print("- combined_fractal_correlation_bw.png (600 DPI, publication quality)")
plt.show()

# Print detailed statistics with enhanced formatting
print("="*80)
print("COMBINED FRACTAL DIMENSION CORRELATION ANALYSIS - RESEARCH SUMMARY")
print("="*80)

print("\n1. CONTROVERSIAL CONVERSATIONS:")
if not richly_branching_data.empty and pearson_r_rich is not None:
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

print("\n2. TECHNICAL CONVERSATIONS:")
if not poorly_branching_data.empty and pearson_r_poor is not None:
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
if (not richly_branching_data.empty and pearson_r_rich is not None and 
    not poorly_branching_data.empty and pearson_r_poor is not None):
    
    summary_data = {
        'Conversation_Type': ['Controversial', 'Technical'],
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
    summary_df.round(4).to_csv('combined_fractal_correlation_summary_table_bw.csv', index=False)
    print("\nSummary table saved as: combined_fractal_correlation_summary_table_bw.csv")

elif not richly_branching_data.empty and pearson_r_rich is not None:
    # Only controversial data available
    summary_data = {
        'Conversation_Type': ['Controversial'],
        'Sample_Size': [len(richly_branching_data)],
        'Pearson_r': [pearson_r_rich],
        'Pearson_p_value': [p_value_rich],
        'Original_Mean_FD': [richly_branching_data['fractal_dimension_original'].mean()],
        'Original_SD_FD': [richly_branching_data['fractal_dimension_original'].std()],
        'Subtree_Mean_FD': [richly_branching_data['fractal_dimension_subtree'].mean()],
        'Subtree_SD_FD': [richly_branching_data['fractal_dimension_subtree'].std()]
    }
    
    summary_df = pd.DataFrame(summary_data)
    print("\n4. RESEARCH SUMMARY TABLE (Controversial Only):")
    print("="*80)
    print(summary_df.round(4).to_string(index=False))
    
elif not poorly_branching_data.empty and pearson_r_poor is not None:
    # Only technical data available
    summary_data = {
        'Conversation_Type': ['Technical'],
        'Sample_Size': [len(poorly_branching_data)],
        'Pearson_r': [pearson_r_poor],
        'Pearson_p_value': [p_value_poor],
        'Original_Mean_FD': [poorly_branching_data['fractal_dimension_original'].mean()],
        'Original_SD_FD': [poorly_branching_data['fractal_dimension_original'].std()],
        'Subtree_Mean_FD': [poorly_branching_data['fractal_dimension_subtree'].mean()],
        'Subtree_SD_FD': [poorly_branching_data['fractal_dimension_subtree'].std()]
    }
    
    summary_df = pd.DataFrame(summary_data)
    print("\n4. RESEARCH SUMMARY TABLE (Technical Only):")
    print("="*80)
    print(summary_df.round(4).to_string(index=False))

print("\n" + "="*80)
print("COMBINED BLACK AND WHITE ANALYSIS COMPLETE - Files ready for research publication")
print("="*80)