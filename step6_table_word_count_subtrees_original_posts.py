import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_analyze_data(original_posts_file, subtree_file):
    """
    Load and analyze conversation data from both original posts and subtrees
    
    Args:
        original_posts_file: Path to the original posts CSV file
        subtree_file: Path to the subtree CSV file
    
    Returns:
        Dictionary containing analysis results and formatted table
    """
    
    # Load the data
    original_posts = pd.read_csv(original_posts_file)
    subtree_data = pd.read_csv(subtree_file)
    
    # Clean column names (strip whitespace)
    original_posts.columns = original_posts.columns.str.strip()
    subtree_data.columns = subtree_data.columns.str.strip()
    
    # Debug: Print available columns
    print("Original Posts columns:", original_posts.columns.tolist())
    print("Subtree data columns:", subtree_data.columns.tolist())
    
    # Create a mapping from post_id to conversation_type from original posts
    if 'post_id' in original_posts.columns and 'post_id' in subtree_data.columns:
        print("Found post_id in both files - creating mapping...")
        
        # Create mapping dictionary
        post_id_to_conversation_type = dict(zip(original_posts['post_id'], original_posts['conversation_type']))
        
        # Add conversation_type to subtree_data based on post_id matching
        subtree_data['conversation_type'] = subtree_data['post_id'].map(post_id_to_conversation_type)
        
        # Check for any unmapped subtrees
        unmapped = subtree_data['conversation_type'].isna().sum()
        if unmapped > 0:
            print(f"Warning: {unmapped} subtrees could not be mapped to parent conversations")
            # Remove unmapped subtrees
            subtree_data = subtree_data.dropna(subset=['conversation_type'])
        
        subtree_type_column = 'conversation_type'
        print(f"Using mapped conversation_type for {len(subtree_data)} subtrees")
        
    elif 'conversation_type' in subtree_data.columns:
        subtree_type_column = 'conversation_type'
        print("Using existing 'conversation_type' for subtrees")
        
    elif 'conversation_type' in subtree_data.columns:
        subtree_type_column = 'conversation_type'
        print("Using 'conversation_type' for subtrees")
        
    else:
        print("Available subtree columns:", subtree_data.columns.tolist())
        raise KeyError("No suitable conversation type column found in subtree data")
    
    # Print the distribution of conversation types
    print("\nOriginal Posts distribution:")
    print(original_posts['conversation_type'].value_counts())
    print(f"\nSubtrees distribution (using {subtree_type_column}):")
    print(subtree_data[subtree_type_column].value_counts())
    
    # Analyze original posts
    original_analysis = analyze_conversation_type(original_posts, 'conversation_type', 'Original Posts')
    
    # Analyze subtrees
    subtree_analysis = analyze_conversation_type(subtree_data, subtree_type_column, 'Subtrees')
    
    # Create comprehensive table
    research_table = create_research_table(original_analysis, subtree_analysis)
    
    # Statistical tests
    statistical_tests = perform_statistical_tests(original_posts, subtree_data, subtree_type_column)
    
    return {
        'original_analysis': original_analysis,
        'subtree_analysis': subtree_analysis,
        'research_table': research_table,
        'statistical_tests': statistical_tests,
        'subtree_type_column': subtree_type_column
    }

def analyze_conversation_type(df, type_column, data_type):
    """
    Analyze conversation data by type (controversial vs technnical branching)
    """
    results = {}
    
    for conv_type in df[type_column].unique():
        subset = df[df[type_column] == conv_type]
        
        # Calculate total words by multiplying total_comments by avg_words_per_comment for each row
        subset_with_total_words = subset.copy()
        subset_with_total_words['total_words'] = subset['total_comments'] * subset['avg_words_per_comment']
        
        results[conv_type] = {
            'data_type': data_type,
            'count': len(subset),
            'total_comments': {
                'sum': subset['total_comments'].sum(),
                'mean': subset['total_comments'].mean(),
                'std': subset['total_comments'].std(),
                'min': subset['total_comments'].min(),
                'max': subset['total_comments'].max()
            },
            'avg_words_per_comment': {
                'mean': subset['avg_words_per_comment'].mean(),
                'std': subset['avg_words_per_comment'].std(),
                'min': subset['avg_words_per_comment'].min(),
                'max': subset['avg_words_per_comment'].max()
            },
            'total_words': {
                'sum': subset_with_total_words['total_words'].sum(),
                'mean': subset_with_total_words['total_words'].mean(),
                'median': subset_with_total_words['total_words'].median(),
                'std': subset_with_total_words['total_words'].std(),
                'min': subset_with_total_words['total_words'].min(),
                'max': subset_with_total_words['total_words'].max()
            }
        }
    
    return results

def create_research_table(original_analysis, subtree_analysis):
    """
    Create a formatted research table for publication
    """
    
    # Define the table structure
    table_data = []
    
    # Original Posts - Controversial Posts
    rich_orig = original_analysis['controversial']
    table_data.append([
        'Controversial Posts',
        f"{rich_orig['total_comments']['sum']:.0f}",
        f"{rich_orig['total_comments']['mean']:.1f}",
        f"{rich_orig['total_words']['sum']:.0f}",
        f"{rich_orig['total_words']['mean']:.0f}",
        f"({rich_orig['total_words']['std']:.0f})",
        f"{rich_orig['avg_words_per_comment']['mean']:.1f}",
        f"({rich_orig['avg_words_per_comment']['std']:.1f})"
    ])
    
    # Original Posts - Technical Posts
    poor_orig = original_analysis['technical']
    table_data.append([
        'Technical Posts',
        f"{poor_orig['total_comments']['sum']:.0f}",
        f"{poor_orig['total_comments']['mean']:.1f}",
        f"{poor_orig['total_words']['sum']:.0f}",
        f"{poor_orig['total_words']['mean']:.0f}",
        f"({poor_orig['total_words']['std']:.0f})",
        f"{poor_orig['avg_words_per_comment']['mean']:.1f}",
        f"({poor_orig['avg_words_per_comment']['std']:.1f})"
    ])
    
    # Subtrees - Richly Branching
    rich_sub = subtree_analysis['controversial']
    table_data.append([
        'Subtrees',
        'Controversial Posts',
        f"{rich_sub['total_comments']['sum']:.0f}",
        f"{rich_sub['total_comments']['mean']:.1f}",
        f"{rich_sub['total_words']['sum']:.0f}",
        f"{rich_sub['total_words']['mean']:.0f}",
        f"({rich_sub['total_words']['std']:.0f})",
        f"{rich_sub['avg_words_per_comment']['mean']:.1f}",
        f"({rich_sub['avg_words_per_comment']['std']:.1f})"
    ])
    
    # Subtrees - Poorly Branching
    poor_sub = subtree_analysis['technical']
    table_data.append([
        'Subtrees',
        'Technical Posts',
        f"{poor_sub['total_comments']['sum']:.0f}",
        f"{poor_sub['total_comments']['mean']:.1f}",
        f"{poor_sub['total_words']['sum']:.0f}",
        f"{poor_sub['total_words']['mean']:.0f}",
        f"({poor_sub['total_words']['std']:.0f})",
        f"{poor_sub['avg_words_per_comment']['mean']:.1f}",
        f"({poor_sub['avg_words_per_comment']['std']:.1f})"
    ])
    
    # Create DataFrame
    columns = [
        'Dataset',
        'Conversation Type',
        'Total Comments',
        'Total Comments(Mean)',
        'Total Words',
        'Total Words(Mean)',
        'Total Words(SD)',
        'Avg Words/Comment(Mean)',
        'Avg Words/Comment(SD)'
    ]
    
    df_table = pd.DataFrame(table_data, columns=columns)
    
    return df_table

def perform_statistical_tests(original_posts, subtree_data, subtree_type_column='conversation_type'):
    """
    Perform statistical tests to compare contoversial vs technical branching conversations
    """
    results = {}
    
    # Original Posts Tests
    rich_orig = original_posts[original_posts['conversation_type'] == 'controversial']
    poor_orig = original_posts[original_posts['conversation_type'] == 'technical']
    
    # Calculate total words for original posts
    rich_orig_total_words = rich_orig['total_comments'] * rich_orig['avg_words_per_comment']
    poor_orig_total_words = poor_orig['total_comments'] * poor_orig['avg_words_per_comment']
    
    # T-tests for original posts
    results['original_posts'] = {
        'total_comments_ttest': stats.ttest_ind(
            rich_orig['total_comments'], 
            poor_orig['total_comments']
        ),
        'avg_words_ttest': stats.ttest_ind(
            rich_orig['avg_words_per_comment'], 
            poor_orig['avg_words_per_comment']
        ),
        'total_words_ttest': stats.ttest_ind(
            rich_orig_total_words,
            poor_orig_total_words
        ),
        'total_comments_mannwhitney': stats.mannwhitneyu(
            rich_orig['total_comments'], 
            poor_orig['total_comments']
        ),
        'avg_words_mannwhitney': stats.mannwhitneyu(
            rich_orig['avg_words_per_comment'], 
            poor_orig['avg_words_per_comment']
        ),
        'total_words_mannwhitney': stats.mannwhitneyu(
            rich_orig_total_words,
            poor_orig_total_words
        )
    }
    
    # Subtree Tests
    rich_sub = subtree_data[subtree_data[subtree_type_column] == 'controversial']
    poor_sub = subtree_data[subtree_data[subtree_type_column] == 'technical']
    
    # Calculate total words for subtrees
    rich_sub_total_words = rich_sub['total_comments'] * rich_sub['avg_words_per_comment']
    poor_sub_total_words = poor_sub['total_comments'] * poor_sub['avg_words_per_comment']
    
    results['subtrees'] = {
        'total_comments_ttest': stats.ttest_ind(
            rich_sub['total_comments'], 
            poor_sub['total_comments']
        ),
        'avg_words_ttest': stats.ttest_ind(
            rich_sub['avg_words_per_comment'], 
            poor_sub['avg_words_per_comment']
        ),
        'total_words_ttest': stats.ttest_ind(
            rich_sub_total_words,
            poor_sub_total_words
        ),
        'total_comments_mannwhitney': stats.mannwhitneyu(
            rich_sub['total_comments'], 
            poor_sub['total_comments']
        ),
        'avg_words_mannwhitney': stats.mannwhitneyu(
            rich_sub['avg_words_per_comment'], 
            poor_sub['avg_words_per_comment']
        ),
        'total_words_mannwhitney': stats.mannwhitneyu(
            rich_sub_total_words,
            poor_sub_total_words
        )
    }
    
    return results

def print_research_table(table_df):
    """
    Print a nicely formatted research table
    """
    print("=" * 140)
    print("CONVERSATION ENGAGEMENT ANALYSIS: CONTROVERSIAL VS TECHNICAL POSTS")
    print("=" * 140)
    print()
    
    # Print table with proper formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print(table_df.to_string(index=False))
    print()
    print("Note: SD = Standard Deviation")
    print("=" * 140)

def print_statistical_summary(statistical_tests):
    """
    Print statistical test results
    """
    print("\nSTATISTICAL TEST RESULTS")
    print("=" * 60)
    
    for dataset, tests in statistical_tests.items():
        print(f"\n{dataset.upper().replace('_', ' ')}")
        print("-" * 30)
        
        # Total Comments
        t_stat, p_val = tests['total_comments_ttest']
        u_stat, p_val_mw = tests['total_comments_mannwhitney']
        print(f"Total Comments:")
        print(f"  t-test: t={t_stat:.3f}, p={p_val:.3f}")
        print(f"  Mann-Whitney U: U={u_stat:.0f}, p={p_val_mw:.3f}")
        
        # Total Words
        t_stat, p_val = tests['total_words_ttest']
        u_stat, p_val_mw = tests['total_words_mannwhitney']
        print(f"Total Words:")
        print(f"  t-test: t={t_stat:.3f}, p={p_val:.3f}")
        print(f"  Mann-Whitney U: U={u_stat:.0f}, p={p_val_mw:.3f}")
        
        # Average Words per Comment
        t_stat, p_val = tests['avg_words_ttest']
        u_stat, p_val_mw = tests['avg_words_mannwhitney']
        print(f"Average Words per Comment:")
        print(f"  t-test: t={t_stat:.3f}, p={p_val:.3f}")
        print(f"  Mann-Whitney U: U={u_stat:.0f}, p={p_val_mw:.3f}")

def create_visualization(original_posts, subtree_data):
    """
    Create visualizations to accompany the research table
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original Posts - Total Comments
    rich_orig = original_posts[original_posts['conversation_type'] == 'controversial posts']
    poor_orig = original_posts[original_posts['conversation_type'] == 'technical posts']
    
    axes[0,0].boxplot([rich_orig['total_comments'], poor_orig['total_comments']], 
                      tick_labels=['Controversial', 'Technical'])
    axes[0,0].set_title('Original Posts: Total Comments')
    axes[0,0].set_ylabel('Total Comments')
    
    # Original Posts - Total Words
    rich_orig_words = rich_orig['total_comments'] * rich_orig['avg_words_per_comment']
    poor_orig_words = poor_orig['total_comments'] * poor_orig['avg_words_per_comment']
    axes[0,1].boxplot([rich_orig_words, poor_orig_words], 
                      tick_labels=['Controversial', 'Technical'])
    axes[0,1].set_title('Original Posts: Total Words')
    axes[0,1].set_ylabel('Total Words')
    
    # Original Posts - Average Words per Comment
    axes[0,2].boxplot([rich_orig['avg_words_per_comment'], poor_orig['avg_words_per_comment']], 
                      tick_labels=['Controversial', 'Technical'])
    axes[0,2].set_title('Original Posts: Average Words per Comment')
    axes[0,2].set_ylabel('Average Words per Comment')
    
    # Determine the correct column for subtrees
    subtree_type_column = 'conversation_type' if 'conversation_type' in subtree_data.columns else 'conversation_type'
    
    # Subtrees - Total Comments
    rich_sub = subtree_data[subtree_data[subtree_type_column] == 'controversial']
    poor_sub = subtree_data[subtree_data[subtree_type_column] == 'technical']
    
    axes[1,0].boxplot([rich_sub['total_comments'], poor_sub['total_comments']], 
                      tick_labels=['Controversial', 'Technical'])
    axes[1,0].set_title('Subtrees: Total Comments')
    axes[1,0].set_ylabel('Total Comments')
    
    # Subtrees - Total Words
    rich_sub_words = rich_sub['total_comments'] * rich_sub['avg_words_per_comment']
    poor_sub_words = poor_sub['total_comments'] * poor_sub['avg_words_per_comment']
    axes[1,1].boxplot([rich_sub_words, poor_sub_words], 
                      tick_labels=['Controversial', 'Technical'])
    axes[1,1].set_title('Subtrees: Total Words')
    axes[1,1].set_ylabel('Total Words')
    
    # Subtrees - Average Words per Comment
    axes[1,2].boxplot([rich_sub['avg_words_per_comment'], poor_sub['avg_words_per_comment']], 
                      tick_labels=['Controversial', 'Technical'])
    axes[1,2].set_title('Subtrees: Average Words per Comment')
    axes[1,2].set_ylabel('Average Words per Comment')
    
    plt.tight_layout()
    plt.show()

# Main execution function
def main():
    """
    Main function to run the complete analysis
    """
    # File paths - adjust these to your actual file locations
    original_posts_file = 'level4_original_posts_word_length_summary_results.csv'
    subtree_file = 'level4_subtree_word_length_summary_results.csv'
    
    # Run analysis
    results = load_and_analyze_data(original_posts_file, subtree_file)
    
    # Print the research table
    print_research_table(results['research_table'])
    
    # Print statistical summary
    print_statistical_summary(results['statistical_tests'])
    
    # Create visualizations
    original_posts = pd.read_csv(original_posts_file)
    subtree_data = pd.read_csv(subtree_file)
    
    # Clean column names for visualization
    original_posts.columns = original_posts.columns.str.strip()
    subtree_data.columns = subtree_data.columns.str.strip()
    
    # Apply the same post_id mapping for visualization if needed
    if 'post_id' in original_posts.columns and 'post_id' in subtree_data.columns:
        post_id_to_conversation_type = dict(zip(original_posts['post_id'], original_posts['conversation_type']))
        subtree_data['conversation_type'] = subtree_data['post_id'].map(post_id_to_conversation_type)
        subtree_data = subtree_data.dropna(subset=['conversation_type'])
    
    create_visualization(original_posts, subtree_data)
    
    # Return the table as a DataFrame for further use
    return results['research_table']

# Example usage
if __name__ == "__main__":
    research_table = main()
