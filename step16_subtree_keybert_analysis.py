import pandas as pd
from keybert import KeyBERT
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
import numpy as np
from collections import defaultdict

# Step 1: Load the CSV file
df = pd.read_csv('subtree_fractal_dimension_replies_analysis.csv')

print("Column names in the CSV:")
print(df.columns.tolist())
print(f"\nDataFrame shape: {df.shape}")

# Check if the required columns exist
required_columns = ['fractal_dimension_type', 'reply']
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    print(f"\nError: Missing columns: {missing_columns}")
    print("Available columns:", df.columns.tolist())
    exit(1)

# Step 2: Group replies by fractal_dimension_type
grouped_data = df.groupby('fractal_dimension_type')['reply'].apply(list).to_dict()

print(f"Found {len(grouped_data)} fractal dimension types:")
for fract_type, replies in grouped_data.items():
    print(f"- {fract_type}: {len(replies)} replies")

# Step 3: Extract keywords for each fractal dimension type
kw_model = KeyBERT(model='all-MiniLM-L6-v2')
all_keywords_by_type = {}

for fract_type, replies in grouped_data.items():
    print(f"\nProcessing {fract_type}...")
    
    # Combine all replies for this fractal dimension type
    combined_text = ' '.join([str(reply) for reply in replies if pd.notna(reply)])
    
    # Extract keywords from the combined text
    if combined_text.strip():  # Only process if there's actual text
        keywords = kw_model.extract_keywords(
            combined_text, 
            keyphrase_ngram_range=(1, 2), 
            stop_words='english', 
            top_n=30  # Increased to get more keywords per type
        )
        # Store keywords with their scores
        all_keywords_by_type[fract_type] = keywords
        print(f"Extracted {len(keywords)} keywords for {fract_type}")
    else:
        all_keywords_by_type[fract_type] = []
        print(f"No valid text found for {fract_type}")

# Step 4: Prepare all keywords for clustering
all_keywords = []
keyword_to_type = {}  # Track which type each keyword came from

for fract_type, keywords in all_keywords_by_type.items():
    for keyword, score in keywords:
        all_keywords.append(keyword)
        if keyword not in keyword_to_type:
            keyword_to_type[keyword] = []
        keyword_to_type[keyword].append((fract_type, score))

# Remove duplicates while preserving the mapping
unique_keywords = list(set(all_keywords))
print(f"\nTotal unique keywords across all types: {len(unique_keywords)}")

# Step 5: Embed keywords
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
keyword_embeddings = embedding_model.encode(unique_keywords)

# Step 6: Determine optimal number of clusters
def find_optimal_clusters(embeddings, max_clusters=15):
    silhouette_scores = []
    cluster_range = range(2, min(max_clusters + 1, len(embeddings)))
    
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        silhouette_scores.append(score)
    
    optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
    return optimal_clusters, max(silhouette_scores)

if len(unique_keywords) > 1:
    optimal_k, best_score = find_optimal_clusters(keyword_embeddings)
    print(f"Optimal number of clusters: {optimal_k} (silhouette score: {best_score:.3f})")
    
    # Step 7: Perform clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(keyword_embeddings)
    
    # Step 8: Group keywords by cluster
    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        keyword = unique_keywords[idx]
        # Include information about which fractal dimension types this keyword came from
        source_types = keyword_to_type[keyword]
        clusters[label].append((keyword, source_types))
    
    # Step 9: Display results
    print("\n" + "="*80)
    print("KEYWORD CLUSTERING RESULTS")
    print("="*80)
    
    for label in sorted(clusters.keys()):
        keywords_info = clusters[label]
        print(f"\nCluster {label} ({len(keywords_info)} keywords):")
        print("-" * 50)
        
        for keyword, source_types in keywords_info:
            # Show which fractal dimension types contributed this keyword
            type_info = []
            for fract_type, score in source_types:
                type_info.append(f"{fract_type}({score:.2f})")
            
            print(f"  • {keyword} -> {', '.join(type_info)}")
    
    # Step 10: Analyze cluster composition by fractal dimension type
    print("\n" + "="*80)
    print("CLUSTER COMPOSITION BY FRACTAL DIMENSION TYPE")
    print("="*80)
    
    type_cluster_distribution = defaultdict(lambda: defaultdict(int))
    
    for label, keywords_info in clusters.items():
        for keyword, source_types in keywords_info:
            for fract_type, score in source_types:
                type_cluster_distribution[fract_type][label] += 1
    
    for fract_type in sorted(type_cluster_distribution.keys()):
        print(f"\n{fract_type}:")
        cluster_counts = type_cluster_distribution[fract_type]
        total_keywords = sum(cluster_counts.values())
        
        for cluster_id in sorted(cluster_counts.keys()):
            count = cluster_counts[cluster_id]
            percentage = (count / total_keywords) * 100
            print(f"  Cluster {cluster_id}: {count} keywords ({percentage:.1f}%)")

else:
    print("Not enough keywords for clustering analysis.")

# Step 11: Save results to files
print("\nSaving results to files...")

# Save keyword extraction results
keyword_results = []
for fract_type, keywords in all_keywords_by_type.items():
    for keyword, score in keywords:
        keyword_results.append({
            'fract_dimension_type': fract_type,
            'keyword': keyword,
            'score': score
        })

keyword_df = pd.DataFrame(keyword_results)
keyword_df.to_csv('subtree_keyword_extraction_results.csv', index=False)
print("Keyword extraction results saved to 'subtree_keyword_extraction_results.csv'")

# Save clustering results
if len(unique_keywords) > 1:
    cluster_results = []
    for label, keywords_info in clusters.items():
        for keyword, source_types in keywords_info:
            for fract_type, score in source_types:
                cluster_results.append({
                    'cluster_id': label,
                    'keyword': keyword,
                    'fract_dimension_type': fract_type,
                    'keyword_score': score
                })
    
    cluster_df = pd.DataFrame(cluster_results)
    cluster_df.to_csv('subtree_clustering_results.csv', index=False)
    print("Clustering results saved to 'subtree_clustering_results.csv'")

print("\nAnalysis complete!")