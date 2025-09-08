import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from scipy import stats

# Create graphs folder if it doesn't exist
graphs = 'graphs'
if not os.path.exists(graphs):
    os.makedirs(graphs)
    print(f"Created folder: {graphs}")

# Set academic publication style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'Arial',
    'axes.linewidth': 1.2,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8
})

# Read the CSV file
df = pd.read_csv('original_posts_time_metrics/time_only_conversation_analysis.csv')

# Separate data by conversation type
richly_branching = df[df['Conversation Type'] == 'richly branching']
poorly_branching = df[df['Conversation Type'] == 'poorly branching']

# Academic colorblind-friendly colors
color_rich = '#1f77b4'  # Blue
color_poor = '#ff7f0e'  # Orange
colors = [color_rich, color_poor]

# Create figure with subplots - 1x2 layout for panels C and D only
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Conversation Structure Analysis: Rich vs. Poor Branching Patterns', 
             fontsize=18, fontweight='bold', y=0.95)

# Calculate sample sizes for display
n_rich = len(richly_branching)
n_poor = len(poorly_branching)

# Panel C: Total Replies Distribution
ax = axes[0]
box_metrics = ['Total Replies']
data_to_plot = [richly_branching['Total Replies'], poorly_branching['Total Replies']]
box_plot = ax.boxplot(data_to_plot, labels=[f'High Branching\n(n={n_rich})', f'Poor Branching\n(n={n_poor})'],
                      patch_artist=True, showmeans=True, meanline=True,
                      boxprops=dict(linewidth=1.2),
                      whiskerprops=dict(linewidth=1.2),
                      capprops=dict(linewidth=1.2),
                      medianprops=dict(linewidth=2, color='black'),
                      meanprops=dict(linewidth=2, color='red', linestyle='--'))

# Color the boxes
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_title('C) Reply Count Distribution', fontweight='bold', pad=30)
ax.set_ylabel('Number of Replies')
ax.grid(True, alpha=0.3, axis='y')

# Panel D: Maximum Depth Distribution
ax = axes[1]
data_to_plot = [richly_branching['Maximum Depth'], poorly_branching['Maximum Depth']]
box_plot = ax.boxplot(data_to_plot, labels=[f'High Branching\n(n={n_rich})', f'Poor Branching\n(n={n_poor})'],
                      patch_artist=True, showmeans=True, meanline=True,
                      boxprops=dict(linewidth=1.2),
                      whiskerprops=dict(linewidth=1.2),
                      capprops=dict(linewidth=1.2),
                      medianprops=dict(linewidth=2, color='black'),
                      meanprops=dict(linewidth=2, color='red', linestyle='--'))

# Color the boxes
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_title('D) Thread Depth Distribution', fontweight='bold', pad=30)
ax.set_ylabel('Maximum Depth (replies)')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.subplots_adjust(top=0.75, bottom=0.15, left=0.1, right=0.95, wspace=0.3)

# Save the first graph with high DPI for publication
graph1_filename = os.path.join(graphs, 'conversation_metrics_comparison_academic.png')
plt.savefig(graph1_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved academic graph: {graph1_filename}")

# Also save as PDF for vector graphics
graph1_pdf = os.path.join(graphs, 'conversation_metrics_comparison_academic.pdf')
plt.savefig(graph1_pdf, bbox_inches='tight', facecolor='white')
print(f"Saved academic graph (PDF): {graph1_pdf}")

plt.show()

# Print detailed comparison statistics with academic formatting
metrics = ['Total Replies', 'Maximum Depth']
print("\nDETAILED STATISTICAL ANALYSIS")
print("=" * 60)
print(f"High Branching Conversations: n = {len(richly_branching)}")
print(f"Poor Branching Conversations: n = {len(poorly_branching)}")
print()

for metric in metrics:
    print(f"{metric.upper().replace(' ', '_')}:")
    print("-" * (len(metric) + 5))
    
    rich_mean = richly_branching[metric].mean()
    rich_std = richly_branching[metric].std()
    rich_median = richly_branching[metric].median()
    rich_sem = richly_branching[metric].sem()
    
    poor_mean = poorly_branching[metric].mean()
    poor_std = poorly_branching[metric].std()
    poor_median = poorly_branching[metric].median()
    poor_sem = poorly_branching[metric].sem()
    
    print(f"High Branching: M = {rich_mean:.2f}, SD = {rich_std:.2f}, Mdn = {rich_median:.2f}, SEM = {rich_sem:.2f}")
    print(f"Poor Branching:  M = {poor_mean:.2f}, SD = {poor_std:.2f}, Mdn = {poor_median:.2f}, SEM = {poor_sem:.2f}")
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(richly_branching)-1)*rich_std**2 + (len(poorly_branching)-1)*poor_std**2) / 
                        (len(richly_branching) + len(poorly_branching) - 2))
    cohens_d = (rich_mean - poor_mean) / pooled_std
    
    print(f"Difference: {rich_mean - poor_mean:.2f} ({((rich_mean - poor_mean)/poor_mean*100):+.1f}%)")
    print(f"Effect size (Cohen's d): {cohens_d:.3f}")
    
    # Statistical significance test (Welch's t-test for unequal variances)
    t_stat, p_value = stats.ttest_ind(richly_branching[metric], poorly_branching[metric], equal_var=False)
    print(f"Welch's t-test: t({len(richly_branching)+len(poorly_branching)-2:.0f}) = {t_stat:.3f}, p = {p_value:.4f}")
    
    # Interpret effect size
    if abs(cohens_d) < 0.2:
        effect_interp = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_interp = "small"
    elif abs(cohens_d) < 0.8:
        effect_interp = "medium"
    else:
        effect_interp = "large"
    
    print(f"Effect size interpretation: {effect_interp}")
    print()

# Create academic-style summary table
print("SUMMARY TABLE FOR MANUSCRIPT:")
print("=" * 80)
print(f"{'Metric':<20} {'High Branch (M±SD)':<20} {'Low Branch (M±SD)':<20} {'p-value':<10} {'Cohen\'s d':<10}")
print("-" * 80)
for metric in ['Total Replies', 'Maximum Depth']:
    rich_mean = richly_branching[metric].mean()
    rich_std = richly_branching[metric].std()
    poor_mean = poorly_branching[metric].mean()
    poor_std = poorly_branching[metric].std()
    
    # Cohen's d calculation
    pooled_std = np.sqrt(((len(richly_branching)-1)*rich_std**2 + (len(poorly_branching)-1)*poor_std**2) / 
                        (len(richly_branching) + len(poorly_branching) - 2))
    cohens_d = (rich_mean - poor_mean) / pooled_std
    
    t_stat, p_value = stats.ttest_ind(richly_branching[metric], poorly_branching[metric], equal_var=False)
    
    print(f"{metric:<20} {rich_mean:.1f}±{rich_std:.1f}{'':>8} {poor_mean:.1f}±{poor_std:.1f}{'':>8} {p_value:.3f}{'':>6} {cohens_d:.2f}")

print(f"\nNote: High branching n={len(richly_branching)}, Low branching n={len(poorly_branching)}")
print("Statistical tests performed using Welch's t-test for unequal variances")

# Create the second graph with academic standards
plt.figure(figsize=(10, 8))
x = np.arange(len(metrics))
width = 0.35

rich_means = [richly_branching[m].mean() for m in metrics]
poor_means = [poorly_branching[m].mean() for m in metrics]
rich_sems = [richly_branching[m].sem() for m in metrics]  # Use SEM instead of SD for error bars
poor_sems = [poorly_branching[m].sem() for m in metrics]

bars1 = plt.bar(x - width/2, rich_means, width, label=f'High Branching (n={n_rich})', 
                color=color_rich, alpha=0.8, yerr=rich_sems, capsize=5, 
                edgecolor='black', linewidth=0.5, error_kw={'linewidth': 2})
bars2 = plt.bar(x + width/2, poor_means, width, label=f'Poor Branching (n={n_poor})', 
                color=color_poor, alpha=0.8, yerr=poor_sems, capsize=5,
                edgecolor='black', linewidth=0.5, error_kw={'linewidth': 2})

plt.xlabel('Conversation Metrics', fontweight='bold', fontsize=14)
plt.ylabel('Mean Values (±SEM)', fontweight='bold', fontsize=14)
plt.title('Comparison of Conversation Structure Metrics', fontweight='bold', fontsize=16, pad=30)
plt.xticks(x, metrics, fontsize=12)
plt.legend(frameon=True, fancybox=False, shadow=False)
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

# Save the second graph
graph2_filename = os.path.join(graphs, 'original_posts_average_metrics_academic.png')
plt.savefig(graph2_filename, dpi=300, bbox_inches='tight', facecolor='white')
graph2_pdf = os.path.join(graphs, 'original_posts_average_metrics_academic.pdf')
plt.savefig(graph2_pdf, bbox_inches='tight', facecolor='white')
print(f"Saved academic graph 2: {graph2_filename}")
print(f"Saved academic graph 2 (PDF): {graph2_pdf}")
plt.show()

print(f"\nAll academic-standard graphs have been saved to the '{graphs}' folder in both PNG and PDF formats.")