from datasketch import MinHash, MinHashLSH
from typing import Dict, List, Set
import networkx as nx
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from Preprocessing import df
from community import community_louvain
import plotly.graph_objects as go
import plotly.express as px  # for colors



def create_shingles(text: str, q: int = 5) -> Set[str]:
    """Convert text into q-grams (shingles)"""
    return {text[i:i+q] for i in range(len(text)-q+1)}

def create_minhash(shingles: Set[str], num_perm: int = 128) -> MinHash:
    """Create MinHash object from shingles"""
    m = MinHash(num_perm=num_perm)
    for shingle in shingles:
        m.update(shingle.encode('utf-8'))
    return m

def extract_top_terms(texts: List[str], n_terms: int = 5) -> List[str]:
    """Extract the most significant terms from a group of tweets using TF-IDF"""
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    avg_scores = np.array(tfidf_matrix.mean(axis=0))[0]
    top_indices = avg_scores.argsort()[-n_terms:][::-1]
    
    feature_names = vectorizer.get_feature_names_out()
    return [feature_names[i] for i in top_indices]

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

def find_topic_clusters(docs: Dict[int, str], 
                       threshold: float = 0.05,
                       num_perm: int = 120,
                       q: int = 5,
                       min_cluster_size: int = 5,
                       n_clusters: int = 10) -> Dict[int, Dict]:
    """
    Find a specified number of topic clusters using LSH and community detection
    
    Args:
        docs: Dictionary of document IDs and their text
        threshold: Jaccard similarity threshold
        num_perm: Number of permutations for MinHash
        q: Size of shingles
        min_cluster_size: Minimum number of documents in a cluster
        n_clusters: Target number of clusters
    
    Returns:
        Dictionary mapping cluster IDs to cluster information
    """
    # Initialize LSH index and similarity graph
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    similarity_graph = nx.Graph()
    
    # Create MinHash objects for all documents
    minhashes = {}
    for doc_id, text in docs.items():
        shingles = create_shingles(text, q)
        minhashes[doc_id] = create_minhash(shingles, num_perm)
        lsh.insert(str(doc_id), minhashes[doc_id])
        similarity_graph.add_node(doc_id)
        print(f"Processed document {doc_id}")

    # Build similarity graph
    processed_pairs = set()
    for doc_id in docs.keys():
        similar = lsh.query(minhashes[doc_id])
        similar = {int(x) for x in similar}
        
        for other_id in similar:
            if doc_id != other_id:
                pair = tuple(sorted([doc_id, other_id]))
                if pair not in processed_pairs:
                    similarity = minhashes[doc_id].jaccard(minhashes[other_id])
                    if threshold <= similarity < 0.40:
                        similarity_graph.add_edge(doc_id, other_id, weight=similarity)
                    processed_pairs.add(pair)

    # Find initial clusters using Louvain
    partition = community_louvain.best_partition(similarity_graph)
    
    # Create cluster dictionary
    clusters = defaultdict(list)
    for doc_id, cluster_id in partition.items():
        clusters[cluster_id].append(doc_id)
    
    # Create initial cluster info
    cluster_info = {}
    for cluster_id, doc_ids in clusters.items():
        if len(doc_ids) >= min_cluster_size:
            cluster_texts = [docs[doc_id] for doc_id in doc_ids]
            cluster_info[cluster_id] = {
                'documents': sorted(doc_ids),
                'size': len(doc_ids),
                'top_terms': extract_top_terms(cluster_texts)
            }
    
    # Merge clusters until reaching target number
    final_clusters = merge_similar_clusters(cluster_info, docs, n_clusters)
    
    return final_clusters







import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import math

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














# Example usage
if __name__ == "__main__":
    # Get documents
    docs = {i: tweet for i, tweet in enumerate(df['tweet'])}
    docs = dict(list(docs.items())[:10000])  # First 1000 tweets
    
    # Find topic clusters
    clusters = find_topic_clusters(
        docs,
        threshold=0.1,
        num_perm=120,
        q=5,
        min_cluster_size=25,
        n_clusters=5  # Specify desired number of clusters
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