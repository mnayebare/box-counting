import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('subtree_sentiment_analysis_level4/tst_sentiment_summary_level4.csv')

# Define sentiment columns and labels
sentiment_cols = ['sentiment_positive_ratio', 'sentiment_negative_ratio', 'sentiment_neutral_ratio']
col_labels = ['Positive', 'Negative', 'Neutral']

# Black and white styling for research papers
colors = ['black', 'white']  # Black for controversial, White for technical

# Calculate medians for each conversation type and sentiment
conversation_types = df['conversation_type'].unique()
medians_data = []

for conv_type in conversation_types:
    subset = df[df['conversation_type'] == conv_type]
    medians = [subset[col].median() for col in sentiment_cols]
    medians_data.append(medians)

# Create the grouped bar chart
fig, ax = plt.subplots(figsize=(10, 6))

# Set width of bars and positions
bar_width = 0.35
x_pos = np.arange(len(col_labels))

# Create bars for each conversation type with black edges
bars1 = ax.bar(x_pos - bar_width/2, medians_data[0], bar_width, 
               label=conversation_types[0].title(), color=colors[0], 
               edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x_pos + bar_width/2, medians_data[1], bar_width,
               label=conversation_types[1].title(), color=colors[1], 
               edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

# Customize the chart
ax.set_xlabel('Sentiment Type', fontsize=12, fontweight='bold')
ax.set_ylabel('Median Sentiment Ratio', fontsize=12, fontweight='bold')
ax.set_title('Median Sentiment Levels in Subtrees by Conversation Type', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(col_labels)
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 0.75)

plt.tight_layout()

# Create graphs folder if it doesn't exist
import os
if not os.path.exists('graphs'):
    os.makedirs('graphs')

# Save the figure as PNG
plt.savefig('graphs/subtrees_median_sentiment_ratios_grouped_bar_chart.png', dpi=300, bbox_inches='tight')
print("Chart saved as: graphs/subtrees_median_sentiment_ratios_grouped_bar_chart.png")

plt.show()

# Print the values used in the chart
print("Median values shown in the chart:")
print("-" * 40)
for i, conv_type in enumerate(conversation_types):
    subset = df[df['conversation_type'] == conv_type]
    print(f"{conv_type.title()} (n={len(subset)}):")
    for j, label in enumerate(col_labels):
        print(f"  {label}: {medians_data[i][j]:.3f}")
    print()