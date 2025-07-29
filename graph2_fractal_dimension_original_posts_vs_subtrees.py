import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_and_prepare_data(original_posts_file, subtree_file):
    """
    Load and prepare data from both CSV files
    """
    # Load the CSV files
    original_df = pd.read_csv(original_posts_file)
    subtree_df = pd.read_csv(subtree_file)
    
    # Add data source labels
    original_df['data_source'] = 'Original Posts'
    subtree_df['data_source'] = 'Subtrees'
    
    # Combine the datasets
    combined_df = pd.concat([original_df, subtree_df], ignore_index=True)
    
    # Create category labels for plotting
    combined_df['category'] = combined_df['data_source'] + '\n' + combined_df['conversation_type'].str.title()
    
    return combined_df, original_df, subtree_df

def create_fractal_dimension_plot(combined_df, save_path=None):
    """
    Create box plot comparing fractal dimensions only - formatted for research paper
    """
    # Set up the plot style for publication
    plt.style.use('default')
    
    # Create figure with appropriate aspect ratio for box plots
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    
    # Create grayscale box plot with white background
    box_plot = sns.boxplot(
        data=combined_df, 
        x='category', 
        y='fractal_dimension',
        ax=ax,
        color='white',  # White background for boxes
        linewidth=1.5
    )
    
    # Customize box plot appearance for publication
    for patch in box_plot.artists:
        patch.set_facecolor('white')  # White background
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
    
    # Customize whiskers, caps, and medians
    for line in box_plot.lines:
        line.set_color('black')
        line.set_linewidth(1.5)
    
    # Overlay individual points in black
    sns.stripplot(
        data=combined_df, 
        x='category', 
        y='fractal_dimension',
        ax=ax,
        size=3,  # Smaller points
        alpha=0.6,
        color='black'
    )
    
    # Customize the plot for publication standards
    ax.set_title('Fractal Dimensions Distribution:\nHighly vs Poorly Branched Conversations', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fractal Dimension', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='black')
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    
    # Add border around the entire plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save with high DPI for publication quality
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    return fig

def print_fractal_summary_statistics(combined_df):
    """
    Print summary statistics for fractal dimensions only
    """
    print("=" * 80)
    print("FRACTAL DIMENSION SUMMARY STATISTICS")
    print("=" * 80)
    
    for category in combined_df['category'].unique():
        data = combined_df[combined_df['category'] == category]
        fractal_data = data['fractal_dimension']
        
        print(f"\n{category}:")
        print(f"  Count: {len(data)}")
        print(f"  Mean: {fractal_data.mean():.4f}")
        print(f"  Median: {fractal_data.median():.4f}")
        print(f"  Std Dev: {fractal_data.std():.4f}")
        print(f"  Range: [{fractal_data.min():.4f}, {fractal_data.max():.4f}]")
        print(f"  Q1: {fractal_data.quantile(0.25):.4f}")
        print(f"  Q3: {fractal_data.quantile(0.75):.4f}")

def compare_groups(combined_df):
    """
    Compare fractal dimensions between different groups
    """
    print("\n" + "=" * 80)
    print("GROUP COMPARISONS")
    print("=" * 80)
    
    # Compare Original Posts vs Subtrees
    original_fractal = combined_df[combined_df['data_source'] == 'Original Posts']['fractal_dimension']
    subtree_fractal = combined_df[combined_df['data_source'] == 'Subtrees']['fractal_dimension']
    
    print(f"\nOriginal Posts vs Subtrees:")
    print(f"  Original Posts - Mean: {original_fractal.mean():.4f}, Std: {original_fractal.std():.4f}")
    print(f"  Subtrees - Mean: {subtree_fractal.mean():.4f}, Std: {subtree_fractal.std():.4f}")
    print(f"  Difference in means: {abs(original_fractal.mean() - subtree_fractal.mean()):.4f}")
    
    # Compare Poorly vs Richly Branching
    poorly_fractal = combined_df[combined_df['conversation_type'] == 'poorly branching']['fractal_dimension']
    richly_fractal = combined_df[combined_df['conversation_type'] == 'richly branching']['fractal_dimension']
    
    print(f"\nPoorly vs Richly Branching:")
    print(f"  Poorly Branching - Mean: {poorly_fractal.mean():.4f}, Std: {poorly_fractal.std():.4f}")
    print(f"  Richly Branching - Mean: {richly_fractal.mean():.4f}, Std: {richly_fractal.std():.4f}")
    print(f"  Difference in means: {abs(poorly_fractal.mean() - richly_fractal.mean()):.4f}")

def main():
    """
    Main function to run the fractal dimension analysis
    """
    # File paths - update these to match your CSV file names
    original_posts_file = "original_posts_fractal_analysis_results.csv"
    subtree_file = "subtree_fractal_analysis.csv"
    
    try:
        # Load and prepare data
        print("Loading data...")
        combined_df, original_df, subtree_df = load_and_prepare_data(original_posts_file, subtree_file)
        
        print(f"Loaded {len(original_df)} original posts and {len(subtree_df)} subtrees")
        print(f"Total records: {len(combined_df)}")
        
        # Create fractal dimension plot
        print("\nCreating fractal dimension comparison plot...")
        fig = create_fractal_dimension_plot(combined_df, save_path="fractal_dimension_comparison.png")
        
        # Print summary statistics
        print_fractal_summary_statistics(combined_df)
        
        # Compare groups
        compare_groups(combined_df)
        
        # Print conversation type distribution
        print("\n" + "=" * 80)
        print("DATA DISTRIBUTION")
        print("=" * 80)
        distribution = combined_df.groupby(['data_source', 'conversation_type']).size().unstack(fill_value=0)
        print(distribution)
        
    except FileNotFoundError as e:
        print(f"Error: Could not find CSV file. Please check the file paths.")
        print(f"Looking for:")
        print(f"  - {original_posts_file}")
        print(f"  - {subtree_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()