from datasketch import MinHash, MinHashLSH
from typing import Dict, List, Set, Tuple
import networkx as nx
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from Preprocessing import df
from community import community_louvain
import plotly.graph_objects as go
import plotly.express as px  # for colors
from concurrent.futures import ThreadPoolExecutor
import itertools
from plotly.subplots import make_subplots
from process_tweet import process_tweet
import math



POLITICAL_STOP_WORDS = [
    'trump', 'biden', 'donald', 'joe', 'potus', '2020', 'trumps'
    'election']

# Combine with English stop words
CUSTOM_STOP_WORDS = list(ENGLISH_STOP_WORDS) + POLITICAL_STOP_WORDS


def create_shingles(text: str, n: int = 2) -> Set[str]:
    """Convert text into n-grams (shingles) at the word level."""
    words = text.split()  # Split the text into words
    if len(words) < n:
        return set()
    return set(" ".join(words[i:i+n]) for i in range(len(words) - n + 1))


def batch_minhash(docs: Dict[int, str], batch_size: int = 1000, num_perm: int = 128, q: int = 5) -> Dict[int, MinHash]:
    """Create MinHash objects in batches"""
    minhashes = {}
    
    def process_batch(batch_items):
        batch_results = {}
        for doc_id, text in batch_items:
            m = MinHash(num_perm=num_perm)
            shingles = create_shingles(text, n=2)
            for shingle in shingles:
                m.update(shingle.encode('utf-8'))
            batch_results[doc_id] = m
        return batch_results
    
    # Process documents in batches
    items = list(docs.items())
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            futures.append(executor.submit(process_batch, batch))
        
        for future in futures:
            minhashes.update(future.result())
    
    return minhashes

def find_candidate_pairs(minhashes: Dict[int, MinHash], threshold: float, 
                        num_bands: int = 24) -> Set[Tuple[int, int]]:
    """Find candidate pairs using LSH with banding technique"""
    n_rows = len(next(iter(minhashes.values())).hashvalues) // num_bands
    candidate_pairs = set()
    
    for band in range(num_bands):
        band_buckets = defaultdict(list)
        
        # Hash document signatures within the current band
        for doc_id, minhash in minhashes.items():
            band_values = tuple(minhash.hashvalues[band * n_rows:(band + 1) * n_rows])
            band_buckets[band_values].append(doc_id)
        
        # Generate candidate pairs from documents in the same bucket
        for bucket in band_buckets.values():
            if len(bucket) > 1:
                candidate_pairs.update(itertools.combinations(sorted(bucket), 2))
    
    return candidate_pairs

def merge_similar_clusters(clusters: Dict[int, Dict], docs: Dict[int, str], target_clusters: int) -> Dict[int, Dict]:
    """Merge similar clusters until reaching target number of clusters"""
    if len(clusters) <= target_clusters:
        return clusters
    
    # Create TF-IDF vectors for each cluster
    vectorizer = TfidfVectorizer(stop_words='english')
    cluster_texts = {}
    for cluster_id, info in clusters.items():
        # Concatenate all texts in cluster
        cluster_text = " ".join([docs[doc_id] for doc_id in info['documents']])
        cluster_texts[cluster_id] = cluster_text
    
    while len(clusters) > target_clusters:
        # Recalculate TF-IDF matrix and similarities for current clusters
        current_cluster_ids = list(clusters.keys())
        tfidf_matrix = vectorizer.fit_transform([cluster_texts[cid] for cid in current_cluster_ids])
        similarities = cosine_similarity(tfidf_matrix)
        
        # Find most similar pair of clusters
        max_similarity = -1
        merge_pair = None
        
        for i in range(len(current_cluster_ids)):
            for j in range(i + 1, len(current_cluster_ids)):
                if similarities[i, j] > max_similarity:
                    max_similarity = similarities[i, j]
                    merge_pair = (current_cluster_ids[i], current_cluster_ids[j])
        
        if merge_pair is None:
            break
            
        # Merge clusters
        cluster1, cluster2 = merge_pair
        new_docs = clusters[cluster1]['documents'] + clusters[cluster2]['documents']
        new_texts = [docs[doc_id] for doc_id in new_docs]
        
        # Update cluster_texts for the merged cluster
        cluster_texts[cluster1] = " ".join([docs[doc_id] for doc_id in new_docs])
        
        # Create new cluster info
        clusters[cluster1] = {
            'documents': new_docs,
            'size': len(new_docs),
            'top_terms': extract_top_terms(new_texts)
        }
        
        # Remove merged cluster and its text
        del clusters[cluster2]
        del cluster_texts[cluster2]
    
    return clusters

def extract_top_terms(texts: List[str], n_terms: int = 5) -> List[str]:
    """Extract the most significant terms from a group of tweets using TF-IDF"""
    vectorizer = TfidfVectorizer(stop_words=CUSTOM_STOP_WORDS, max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    avg_scores = np.array(tfidf_matrix.mean(axis=0))[0]
    top_indices = avg_scores.argsort()[-n_terms:][::-1]
    
    feature_names = vectorizer.get_feature_names_out()
    return [feature_names[i] for i in top_indices]

def find_topic_clusters(docs: Dict[int, str], 
                       threshold: float = 0.05,
                       num_perm: int = 120,
                       q: int = 4,
                       min_cluster_size: int = 5,
                       n_clusters: int = 10) -> Dict[int, Dict]:
    """
    Optimized version of topic clustering using LSH and banding technique
    """
    # Initialize similarity graph
    similarity_graph = nx.Graph()
    similarity_graph.add_nodes_from(docs.keys())
    
    # Create MinHash signatures in batches
    print("Creating MinHash signatures...")
    minhashes = batch_minhash(docs, batch_size=1000, num_perm=num_perm, q=q)
    
    # Find candidate pairs using LSH banding
    print("Finding candidate pairs...")
    candidate_pairs = find_candidate_pairs(minhashes, threshold)
    
    # Verify candidate pairs and build similarity graph
    print("Building similarity graph...")
    for doc_id1, doc_id2 in candidate_pairs:
        similarity = minhashes[doc_id1].jaccard(minhashes[doc_id2])
        if threshold <= similarity < 0.40:
            similarity_graph.add_edge(doc_id1, doc_id2, weight=similarity)
    
    # Find clusters using Louvain
    print("Finding communities...")
    partition = community_louvain.best_partition(similarity_graph)
    
    # Create cluster dictionary
    clusters = defaultdict(list)
    for doc_id, cluster_id in partition.items():
        clusters[cluster_id].append(doc_id)
    
    # Create cluster info with batched TF-IDF calculation
    print("Creating cluster information...")
    cluster_info = {}
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    
    # Process clusters that meet minimum size requirement
    valid_clusters = {cid: doc_ids for cid, doc_ids in clusters.items() 
                     if len(doc_ids) >= min_cluster_size}
    
    if valid_clusters:
        # Prepare texts for all valid clusters at once
        all_texts = []
        cluster_doc_mapping = []
        for cluster_id, doc_ids in valid_clusters.items():
            cluster_texts = [docs[doc_id] for doc_id in doc_ids]
            all_texts.extend(cluster_texts)
            cluster_doc_mapping.extend([cluster_id] * len(cluster_texts))
        
        # Calculate TF-IDF for all texts at once
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Process each cluster
        current_idx = 0
        for cluster_id, doc_ids in valid_clusters.items():
            n_docs = len(doc_ids)
            cluster_tfidf = tfidf_matrix[current_idx:current_idx + n_docs]
            
            # Calculate average TF-IDF scores for the cluster
            avg_scores = np.array(cluster_tfidf.mean(axis=0))[0]
            top_indices = avg_scores.argsort()[-5:][::-1]  # Get top 5 terms
            
            cluster_info[cluster_id] = {
                'documents': sorted(doc_ids),
                'size': n_docs,
                'top_terms': [feature_names[i] for i in top_indices]
            }
            current_idx += n_docs
    
    # Merge similar clusters if needed
    if len(cluster_info) > n_clusters:
        cluster_info = merge_similar_clusters(cluster_info, docs, n_clusters)
    
    return cluster_info

def create_cluster_word_viz(clusters, max_terms=10):
    """
    Create a visualization where each cluster is represented by its key terms
    with the most significant term larger in the center
    
    Args:
        clusters: Dictionary of clusters from find_topic_clusters()
        max_terms: Maximum number of terms to show per cluster
    """
    # Calculate grid dimensions
    n_clusters = len(clusters)
    n_cols = min(3, n_clusters)  # Maximum 3 columns
    n_rows = math.ceil(n_clusters / n_cols)
    
    # Create figure with subplots
    fig = make_subplots(
        rows=n_rows, 
        cols=n_cols,
        subplot_titles=[f'Cluster {i}' for i in range(n_clusters)],
        vertical_spacing=0.2,
        horizontal_spacing=0.1
    )
    
    # Colors for different clusters
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for idx, (cluster_id, info) in enumerate(clusters.items()):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        color = colors[idx % len(colors)]
        
        # Get top terms (limit to max_terms)
        terms = info['top_terms'][:max_terms]
        
        # Calculate positions for terms in a circular pattern
        n_outer_terms = len(terms) - 1  # All terms except the main one
        radius = 0.4  # Radius for outer terms
        
        # Center position for the main term
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                text=[terms[0]],
                mode='text',
                textfont=dict(
                    size=40,  # Larger size for main term
                    color=color,
                    family='Arial Black'
                ),
                hoverinfo='text',
                hovertext=f'Main topic: {terms[0]}',
                showlegend=False
            ),
            row=row,
            col=col
        )
        
        # Add size indicator
        fig.add_annotation(
            text=f'Size: {info["size"]} documents',
            xref=f'x{idx+1}',
            yref=f'y{idx+1}',
            x=0,
            y=1,
            showarrow=False,
            font=dict(size=12),
            row=row,
            col=col
        )
        
        # Position other terms in a circle around the main term
        if n_outer_terms > 0:
            for i, term in enumerate(terms[1:], 1):
                angle = (2 * np.pi * (i-1) / n_outer_terms) - (np.pi/2)
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                
                # Calculate font size based on position in list
                font_size = max(15, 30 - (i * 2))  # Decreasing size for less important terms
                
                fig.add_trace(
                    go.Scatter(
                        x=[x],
                        y=[y],
                        text=[term],
                        mode='text',
                        textfont=dict(
                            size=font_size,
                            color=color,
                            family='Arial'
                        ),
                        hoverinfo='text',
                        hovertext=f'Term: {term}',
                        showlegend=False
                    ),
                    row=row,
                    col=col
                )
    
    # Update layout
    fig.update_layout(
        title_text="Topic Clusters - Most Significant Terms",
        title_x=0.5,
        title_font_size=24,
        height=400 * n_rows,
        width=1000,
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    # Update axes for all subplots
    for i in range(1, n_clusters + 1):
        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 1], row=(i-1)//n_cols + 1, col=(i-1)%n_cols + 1)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 1], row=(i-1)//n_cols + 1, col=(i-1)%n_cols + 1)
    
    return fig

def analyze_cluster_overlap(clusters, docs):
    """Analyze term overlap between clusters to check distinctness"""
    all_terms = set()
    overlap_count = defaultdict(int)
    
    for cluster_id, info in clusters.items():
        terms = set(info['top_terms'])
        # Check overlap with existing terms
        overlap = terms & all_terms
        if overlap:
            print(f"\nCluster {cluster_id} shares terms with other clusters:")
            print(f"Overlapping terms: {', '.join(overlap)}")
        all_terms.update(terms)
        
        # Count term frequencies across all documents in cluster
        vectorizer = TfidfVectorizer(stop_words='english')
        cluster_docs = [docs[doc_id] for doc_id in info['documents']]
        tfidf = vectorizer.fit_transform(cluster_docs)
        avg_tfidf = np.array(tfidf.mean(axis=0))[0]
        
        print(f"\nCluster {cluster_id} coherence:")
        print(f"Average document similarity: {avg_tfidf.mean():.3f}")
        print(f"Unique terms ratio: {len(set(info['top_terms']))/5:.2f}")

# Example usage
if __name__ == "__main__":
    # Get documents
    docs = {i: tweet for i, tweet in enumerate(df['tweet'])}
    docs = dict(list(docs.items())[:50000])  # First 1000 tweets
    print("Processing hashtags...")
    for i, tweet in enumerate(docs.values()):
        docs[i] = process_tweet(tweet)
    
    # Find topic clusters
    clusters = find_topic_clusters(
        docs,
        threshold=0.2, #0.15
        num_perm=120,
        q=4,
        min_cluster_size=30,
        n_clusters=5
    )
    
    # Print cluster information
    print(f"\nFound {len(clusters)} topic clusters:")
    for cluster_id, info in clusters.items():
        print(f"\nCluster {cluster_id}:")
        print(f"Size: {info['size']} documents")
        print(f"Top terms: {', '.join(info['top_terms'])}")
        print("\nExample documents:")
        for doc_id in info['documents'][:3]:
            print(f"- {docs[doc_id][:100]}...")
        print("---")

    # # Example usage:
    fig = create_cluster_word_viz(clusters, 4)
    fig.show()
    analyze_cluster_overlap(clusters, docs)