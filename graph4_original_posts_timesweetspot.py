import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# Clear any matplotlib cache
plt.clf()
plt.close('all')

# Clear variables to avoid cache issues
globals().pop('rb_binned_depths', None)
globals().pop('pb_binned_depths', None)
globals().pop('rb_bin_counts', None)
globals().pop('pb_bin_counts', None)
globals().pop('rb_std_errors', None)
globals().pop('pb_std_errors', None)

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

# Load the data
df = pd.read_csv('original_posts_time_metrics/time_only_conversation_analysis.csv')

# Clean column names
df.columns = df.columns.str.strip()

# Separate by conversation type
richly_branching = df[df['Conversation Type'] == 'richly branching']
poorly_branching = df[df['Conversation Type'] == 'poorly branching']

# Extract 80% engagement timing data (removing nulls)
rb_80 = richly_branching['Time to 80% Engagement (hours)'].dropna()
pb_80 = poorly_branching['Time to 80% Engagement (hours)'].dropna()

# Extract depth data - use the Maximum Depth column
print(f"Available columns: {list(df.columns)}")

# Use the Maximum Depth column
depth_column = 'Maximum Depth'
if depth_column not in df.columns:
    print(f"ERROR: '{depth_column}' column not found. Available columns:")
    for col in df.columns:
        print(f"  - {col}")
    raise ValueError(f"'{depth_column}' column not found in the data. Please check your CSV file.")

print(f"Using depth column: {depth_column}")
rb_depth = richly_branching[depth_column].dropna()
pb_depth = poorly_branching[depth_column].dropna()

# Ensure we have matching time and depth data
# Align the arrays to same length
min_len_rb = min(len(rb_80), len(rb_depth))
min_len_pb = min(len(pb_80), len(pb_depth))

rb_time = rb_80.iloc[:min_len_rb].reset_index(drop=True)
rb_depth_aligned = rb_depth.iloc[:min_len_rb].reset_index(drop=True)
pb_time = pb_80.iloc[:min_len_pb].reset_index(drop=True)
pb_depth_aligned = pb_depth.iloc[:min_len_pb].reset_index(drop=True)

print("=== DATA SUMMARY ===")
print(f"Rich Branching - 80% Engagement: {len(rb_time)} conversations")
print(f"Poor Branching - 80% Engagement: {len(pb_time)} conversations")
print(f"Rich Time - Mean: {rb_time.mean():.1f}h, Median: {rb_time.median():.1f}h")
print(f"Poor Time - Mean: {pb_time.mean():.1f}h, Median: {pb_time.median():.1f}h")
print(f"Rich Depth - Mean: {rb_depth_aligned.mean():.1f}, Max: {rb_depth_aligned.max()}")
print(f"Poor Depth - Mean: {pb_depth_aligned.mean():.1f}, Max: {pb_depth_aligned.max()}")

# Academic colorblind-friendly colors
colors = {
    'rich_branching': '#1f77b4',  # Blue
    'poor_branching': '#ff7f0e',  # Orange
    'sweet_spot': '#2ca02c'       # Green
}

# Set up the plotting with proper academic layout
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Conversation Depth vs. Time Analysis', 
             fontsize=18, fontweight='bold', y=0.98)

# Calculate sample sizes for display
n_rich = len(rb_time)
n_poor = len(pb_time)

# 1. Bar Graph - Depth by Time Bins (Left) - Panel A
# Create time bins for grouping
time_bins = np.arange(0, 50, 4)  # 4-hour bins
bin_centers = time_bins[:-1] + 2  # Center of each bin
bin_labels = [f'{int(time_bins[i])}-{int(time_bins[i+1])}h' for i in range(len(time_bins)-1)]

# Calculate mean depth for each time bin
rb_binned_depths = []
pb_binned_depths = []

for i in range(len(time_bins)-1):
    # Rich branching
    mask_rb = (rb_time >= time_bins[i]) & (rb_time < time_bins[i+1])
    if mask_rb.sum() > 0:
        rb_binned_depths.append(rb_depth_aligned[mask_rb].mean())
    else:
        rb_binned_depths.append(0)
    
    # Poor branching
    mask_pb = (pb_time >= time_bins[i]) & (pb_time < time_bins[i+1])
    if mask_pb.sum() > 0:
        pb_binned_depths.append(pb_depth_aligned[mask_pb].mean())
    else:
        pb_binned_depths.append(0)

# Create bar positions
x_pos = np.arange(len(bin_centers))
bar_width = 0.35

# Create side-by-side bars
bars1 = ax1.bar(x_pos - bar_width/2, rb_binned_depths, bar_width, 
                label='Rich Branching', 
                color=colors['rich_branching'], alpha=0.8, 
                edgecolor='black', linewidth=0.5)

bars2 = ax1.bar(x_pos + bar_width/2, pb_binned_depths, bar_width, 
                label='Poor Branching', 
                color=colors['poor_branching'], alpha=0.8, 
                edgecolor='black', linewidth=0.5)

ax1.set_xlabel('Time (hours)', fontweight='bold')
ax1.set_ylabel('Average Depth (levels)', fontweight='bold')  # Simplified from "Mean Conversation Depth (levels)"
ax1.set_title('A) Depth vs Time', fontweight='bold', pad=20)
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f'{int(time_bins[i])}-{int(time_bins[i+1])}' for i in range(len(time_bins)-1)], rotation=45)
ax1.legend(frameon=True, fancybox=False, shadow=False)
ax1.grid(True, alpha=0.3, axis='y')

# 2. Cumulative Coverage Analysis (Right) - Panel B
time_points = np.arange(1, 49, 1)

# Calculate cumulative coverage for each time point
rb_80_coverage = []
pb_80_coverage = []

for t in time_points:
    rb_80_coverage.append(100 * (rb_time <= t).sum() / len(rb_time))
    pb_80_coverage.append(100 * (pb_time <= t).sum() / len(pb_time))

# Plot with academic styling
line1 = ax2.plot(time_points, rb_80_coverage, label='Rich Branching', 
                 color=colors['rich_branching'], linewidth=3, marker='o', markersize=3, 
                 markevery=4, alpha=0.8)
line2 = ax2.plot(time_points, pb_80_coverage, label='Poor Branching', 
                 color=colors['poor_branching'], linewidth=3, marker='s', markersize=3, 
                 markevery=4, alpha=0.8)

ax2.set_xlabel('Time (hours)', fontweight='bold')
ax2.set_ylabel('Conversations Captured (%)', fontweight='bold')
ax2.set_title('B) Engagement Capture', fontweight='bold', pad=20)
ax2.legend(frameon=True, fancybox=False, shadow=False)
ax2.grid(True, alpha=0.3)

# Find optimal sweet spot (where both curves have reasonable coverage)
# Look for point where both types have at least 60% coverage
optimal_times = []
for i, t in enumerate(time_points):
    if rb_80_coverage[i] >= 60 and pb_80_coverage[i] >= 60:
        optimal_times.append(t)

if optimal_times:
    sweet_spot = optimal_times[0]  # First time point where both reach 60%
    
    # Add sweet spot line with academic styling
    ax2.axvline(x=sweet_spot, color=colors['sweet_spot'], linestyle=':', 
                linewidth=3, alpha=0.9)
    
    rb_coverage_at_spot = 100 * (rb_time <= sweet_spot).sum() / len(rb_time)
    pb_coverage_at_spot = 100 * (pb_time <= sweet_spot).sum() / len(pb_time)
    
    # Clean annotation without clutter
    ax2.annotate(f'Sweet Spot\n{sweet_spot}h', 
                 xy=(sweet_spot, 70), xytext=(sweet_spot + 8, 50),
                 arrowprops=dict(arrowstyle='->', color=colors['sweet_spot'], lw=2),
                 fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='white', 
                          edgecolor=colors['sweet_spot'], alpha=0.8))
else:
    # Fallback to median of medians
    sweet_spot = int((rb_time.median() + pb_time.median()) / 2)
    ax2.axvline(x=sweet_spot, color=colors['sweet_spot'], linestyle=':', 
                linewidth=3, alpha=0.9)

# Set proper y-axis limits and ticks
ax2.set_yticks(np.arange(0, 101, 20))
ax2.set_ylim(0, 100)
ax2.set_xlim(0, 48)

# Improve overall layout with better spacing and padding
plt.tight_layout()
plt.subplots_adjust(top=0.88, bottom=0.15)

# Save the graph with academic standards and proper padding
graph_filename = os.path.join(graphs, 'original_posts_depth_vs_time_bar_chart.png')
plt.savefig(graph_filename, dpi=300, bbox_inches='tight', facecolor='white', 
            pad_inches=0.3)
graph_pdf = os.path.join(graphs, 'original_posts_depth_vs_time_bar_chart.pdf') 

plt.savefig(graph_pdf, bbox_inches='tight', facecolor='white', 
            pad_inches=0.3)

print(f"Saved graph: {graph_filename}")
print(f"Saved graph (PDF): {graph_pdf}")
plt.show()

# Print detailed academic analysis
print("\n" + "="*60)
print("STATISTICAL ANALYSIS: DEPTH vs TIME")
print("="*60)

# Statistical comparison
print(f"\nDESCRIPTIVE STATISTICS:")
print(f"Rich Branching (n={len(rb_time)}):")
print(f"  Time - Mean ± SD: {rb_time.mean():.2f} ± {rb_time.std():.2f} hours")
print(f"  Time - Median (IQR): {rb_time.median():.1f} ({rb_time.quantile(0.25):.1f}-{rb_time.quantile(0.75):.1f}) hours")
print(f"  Depth - Mean ± SD: {rb_depth_aligned.mean():.2f} ± {rb_depth_aligned.std():.2f} levels")
print(f"  Depth - Median (IQR): {rb_depth_aligned.median():.1f} ({rb_depth_aligned.quantile(0.25):.1f}-{rb_depth_aligned.quantile(0.75):.1f}) levels")

print(f"\nPoor Branching (n={len(pb_time)}):")
print(f"  Time - Mean ± SD: {pb_time.mean():.2f} ± {pb_time.std():.2f} hours")
print(f"  Time - Median (IQR): {pb_time.median():.1f} ({pb_time.quantile(0.25):.1f}-{pb_time.quantile(0.75):.1f}) hours")
print(f"  Depth - Mean ± SD: {pb_depth_aligned.mean():.2f} ± {pb_depth_aligned.std():.2f} levels")
print(f"  Depth - Median (IQR): {pb_depth_aligned.median():.1f} ({pb_depth_aligned.quantile(0.25):.1f}-{pb_depth_aligned.quantile(0.75):.1f}) levels")

# Statistical testing for time differences
t_stat_time, p_val_time = stats.ttest_ind(rb_time, pb_time, equal_var=False)
u_stat_time, p_val_mw_time = stats.mannwhitneyu(rb_time, pb_time, alternative='two-sided')

# Statistical testing for depth differences
t_stat_depth, p_val_depth = stats.ttest_ind(rb_depth_aligned, pb_depth_aligned, equal_var=False)
u_stat_depth, p_val_mw_depth = stats.mannwhitneyu(rb_depth_aligned, pb_depth_aligned, alternative='two-sided')

# Effect sizes
pooled_std_time = np.sqrt(((len(rb_time)-1)*rb_time.std()**2 + (len(pb_time)-1)*pb_time.std()**2) / 
                         (len(rb_time) + len(pb_time) - 2))
cohens_d_time = (rb_time.mean() - pb_time.mean()) / pooled_std_time

pooled_std_depth = np.sqrt(((len(rb_depth_aligned)-1)*rb_depth_aligned.std()**2 + (len(pb_depth_aligned)-1)*pb_depth_aligned.std()**2) / 
                          (len(rb_depth_aligned) + len(pb_depth_aligned) - 2))
cohens_d_depth = (rb_depth_aligned.mean() - pb_depth_aligned.mean()) / pooled_std_depth

print(f"\nSTATISTICAL TESTS:")
print(f"Time Comparison:")
print(f"  Welch's t-test: t = {t_stat_time:.3f}, p = {p_val_time:.4f}")
print(f"  Mann-Whitney U test: U = {u_stat_time:.0f}, p = {p_val_mw_time:.4f}")
print(f"  Effect size (Cohen's d): {cohens_d_time:.3f}")

print(f"\nDepth Comparison:")
print(f"  Welch's t-test: t = {t_stat_depth:.3f}, p = {p_val_depth:.4f}")
print(f"  Mann-Whitney U test: U = {u_stat_depth:.0f}, p = {p_val_mw_depth:.4f}")
print(f"  Effect size (Cohen's d): {cohens_d_depth:.3f}")

# Correlation analysis
corr_rb = np.corrcoef(rb_time, rb_depth_aligned)[0, 1]
corr_pb = np.corrcoef(pb_time, pb_depth_aligned)[0, 1]

print(f"\nCORRELATION ANALYSIS (Depth vs Time):")
print(f"Rich Branching: r = {corr_rb:.4f}")
corr_p_rb = stats.pearsonr(rb_time, rb_depth_aligned)[1]
print(f"  p-value: {corr_p_rb:.4f}")

print(f"Poor Branching: r = {corr_pb:.4f}")
corr_p_pb = stats.pearsonr(pb_time, pb_depth_aligned)[1]
print(f"  p-value: {corr_p_pb:.4f}")

# Coverage analysis with depth information
print(f"\nCOVERAGE ANALYSIS:")
key_times = [12, 18, 24, 36, 48]
print(f"{'Time':<6} {'Rich (%)':<10} {'Poor (%)':<10} {'Min (%)':<8}")
print("-" * 40)
for t in key_times:
    rb_cov = 100 * (rb_time <= t).sum() / len(rb_time)
    pb_cov = 100 * (pb_time <= t).sum() / len(pb_time)
    min_cov = min(rb_cov, pb_cov)
    print(f"{t:2d}h    {rb_cov:6.1f}     {pb_cov:6.1f}     {min_cov:5.1f}")

# Depth analysis by time bins
print(f"\nDEPTH BY TIME BINS:")
time_bins = [(0, 12), (12, 24), (24, 36), (36, 48)]
print(f"{'Time Bin':<12} {'Rich Depth':<12} {'Poor Depth':<12} {'Rich n':<8} {'Poor n':<8}")
print("-" * 64)

for start, end in time_bins:
    rb_mask = (rb_time >= start) & (rb_time < end)
    pb_mask = (pb_time >= start) & (pb_time < end)
    
    rb_depth_bin = rb_depth_aligned[rb_mask]
    pb_depth_bin = pb_depth_aligned[pb_mask]
    
    rb_mean = rb_depth_bin.mean() if len(rb_depth_bin) > 0 else 0
    pb_mean = pb_depth_bin.mean() if len(pb_depth_bin) > 0 else 0
    
    print(f"{start:2d}-{end:2d}h      {rb_mean:8.1f}     {pb_mean:8.1f}     {len(rb_depth_bin):4d}     {len(pb_depth_bin):4d}")

if 'sweet_spot' in locals():
    print(f"\n🎯 OPTIMAL DATA COLLECTION WINDOW: {sweet_spot} HOURS")
    print(f"At {sweet_spot}h, captures:")
    print(f"  • {100 * (rb_time <= sweet_spot).sum() / len(rb_time):.0f}% of rich branching conversations")
    print(f"  • {100 * (pb_time <= sweet_spot).sum() / len(pb_time):.0f}% of poor branching conversations")
    print(f"  • Balanced coverage for both conversation types")
else:
    print(f"\n🎯 RECOMMENDED COLLECTION WINDOW: 24 HOURS")
    print("Provides practical balance for capturing 80% engagement data.")

# Create summary table for manuscript
print(f"\nSUMMARY TABLE FOR MANUSCRIPT:")
print("=" * 80)
print(f"{'Metric':<30} {'Rich Branch':<15} {'Poor Branch':<15} {'p-value':<10}")
print("-" * 80)
print(f"{'Sample size':<30} {len(rb_time):<15} {len(pb_time):<15} {'-':<10}")
print(f"{'Time Mean ± SD (hours)':<30} {rb_time.mean():.1f} ± {rb_time.std():.1f}{'':>4} {pb_time.mean():.1f} ± {pb_time.std():.1f}{'':>4} {p_val_time:.3f}")
print(f"{'Time Median (IQR) (hours)':<30} {rb_time.median():.1f} ({rb_time.quantile(0.25):.1f}-{rb_time.quantile(0.75):.1f}){'':>2} {pb_time.median():.1f} ({pb_time.quantile(0.25):.1f}-{pb_time.quantile(0.75):.1f}){'':>2} {p_val_mw_time:.3f}")
print(f"{'Depth Mean ± SD (levels)':<30} {rb_depth_aligned.mean():.1f} ± {rb_depth_aligned.std():.1f}{'':>4} {pb_depth_aligned.mean():.1f} ± {pb_depth_aligned.std():.1f}{'':>4} {p_val_depth:.3f}")
print(f"{'Depth-Time Correlation (r)':<30} {corr_rb:.3f}{'':>10} {corr_pb:.3f}{'':>10} {'-':<10}")
print(f"{'Time Effect Size (Cohen\'s d)':<30} {cohens_d_time:.2f}{'':>10} {'-':<15} {'-':<10}")
print(f"{'Depth Effect Size (Cohen\'s d)':<30} {cohens_d_depth:.2f}{'':>10} {'-':<15} {'-':<10}")

print(f"\nNote: Statistical tests performed using Welch's t-test and Mann-Whitney U test")