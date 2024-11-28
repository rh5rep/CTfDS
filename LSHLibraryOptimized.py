from datasketch import MinHash, MinHashLSH
from typing import Dict, List, Set, Tuple
import networkx as nx
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from Preprocessing import df, process_tweet
from community import community_louvain
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor
from plotly.subplots import make_subplots
import math

POLITICAL_STOP_WORDS = [
    'trump', 'biden', 'donald', 'joe', 'potus', '2020', 'trumps', 'realdonaldtrump', 'harris',
    'save', 'yep', 'yes', 'nope', 'no', 'fa', 'oh', 'nytimes', 'wow', 'tells', 'omg', 'wait', 'look', 'abc'
    'abcnews', 'cnn', 'fox', 'foxnews', 'msnbc', 'nbc', 'nbcnews', 'cbs', 'cbsnews', 'news', 'joebiden', 'did', 'won', 
    'wins', '19', 'kamala', 'wouldn', 'rawstory']
CUSTOM_STOP_WORDS = list(ENGLISH_STOP_WORDS) + POLITICAL_STOP_WORDS
MAX_FEATURES = 3000


def create_vectorizer(max_features: int = MAX_FEATURES) -> TfidfVectorizer:
    """Create a consistent TfidfVectorizer with standard parameters"""
    return TfidfVectorizer(
        stop_words=CUSTOM_STOP_WORDS,
        max_features=max_features,
        ngram_range=(1, 1),  # unigrams
        min_df=7,  # minimum document frequency
        max_df=0.7  # maximum document frequency
    )

def create_shingles(text: str, q: int = 5) -> Set[str]:
    """Convert text into q-grams (shingles)"""
    # Process only if text is long enough
    if len(text) < q:
        return set()
    return set(text[i:i+q] for i in range(len(text)-q+1))

def batch_minhash(docs: Dict[int, str], batch_size: int = 1000, num_perm: int = 128, q: int = 5) -> Dict[int, MinHash]:
    """Create MinHash objects in batches"""
    minhashes = {}
    
    def process_batch(batch_items):
        batch_results = {}
        for doc_id, text in batch_items:
            m = MinHash(num_perm=num_perm)
            shingles = create_shingles(text, q)
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

def find_candidate_pairs(minhashes: Dict[int, MinHash], threshold: float) -> Set[Tuple[int, int]]:
    """
    Memory-efficient candidate pair generation using LSH with iterative processing
    
    Args:
        minhashes: Dictionary of document IDs to MinHash signatures
        threshold: Jaccard similarity threshold for considering pairs
    
    Returns:
        Set of document ID pairs that are potentially similar
    """
    # Limit memory by processing documents in batches
    candidate_pairs = set()
    doc_ids = list(minhashes.keys())
    
    # Process in manageable chunks to reduce memory pressure
    for i in range(0, len(doc_ids), 10000):  # Adjust batch size as needed
        batch_doc_ids = doc_ids[i:i+10000]
        
        # Create a temporary LSH index for this batch
        lsh = MinHashLSH(threshold=threshold, num_perm=len(next(iter(minhashes.values())).hashvalues))
        
        # Insert documents from this batch into LSH
        for doc_id in batch_doc_ids:
            lsh.insert(doc_id, minhashes[doc_id])
        
        # Find pairs within this batch and across previous batches
        for doc_id in batch_doc_ids:
            similar_ids = lsh.query(minhashes[doc_id])
            
            for similar_id in similar_ids:
                if similar_id != doc_id:
                    # Ensure consistent pair ordering
                    candidate_pair = tuple(sorted((doc_id, similar_id)))
                    candidate_pairs.add(candidate_pair)
        
        # Clear LSH to free memory
        del lsh
    
    return candidate_pairs


def merge_similar_clusters(clusters: Dict[int, Dict], docs: Dict[int, str], target_clusters: int, n_terms: int) -> Dict[int, Dict]:
    """Merge similar clusters until reaching target number of clusters"""
    if len(clusters) <= target_clusters:
        return clusters
    
    # Create TF-IDF vectors for each cluster
    vectorizer = TfidfVectorizer(stop_words=CUSTOM_STOP_WORDS)
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
            'top_terms': extract_top_terms(new_texts, n_terms=n_terms)
        }
        
        # Remove merged cluster and its text
        del clusters[cluster2]
        del cluster_texts[cluster2]
    
    return clusters

def extract_top_terms(texts: List[str], n_terms: int = 5) -> List[str]:
    """Extract the most significant terms from a group of tweets using TF-IDF"""
    vectorizer = create_vectorizer()
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
                       n_terms: int = 5,
                       batch_size: int = 1000,
                       n_clusters: int = 10,
                       max_docs: int = None) -> Dict[int, Dict]:
    """
    Optimized version with memory-conscious processing
    """
    # Optionally limit number of documents
    if max_docs is not None and len(docs) > max_docs:
        print(f"Warning: Large dataset detected. Using first {max_docs} documents.")
        docs = dict(list(docs.items())[:max_docs])

    # Initialize similarity graph
    similarity_graph = nx.Graph()
    similarity_graph.add_nodes_from(docs.keys())
    
    # Create MinHash signatures in batches
    print("Creating MinHash signatures...")
    minhashes = batch_minhash(docs, batch_size=batch_size, num_perm=num_perm, q=q)
    
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
    vectorizer = create_vectorizer()
    
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
            top_indices = avg_scores.argsort()[-n_terms:][::-1]  # Get top n_terms
            
            cluster_info[cluster_id] = {
                'documents': sorted(doc_ids),
                'size': n_docs,
                'top_terms': [feature_names[i] for i in top_indices]
            }
            current_idx += n_docs
    
    # Merge similar clusters if needed
    if len(cluster_info) > n_clusters:
        cluster_info = merge_similar_clusters(cluster_info, docs, n_clusters, n_terms)
    
    return cluster_info

def create_cluster_wordclouds(clusters, max_terms=10):
    """
    Create a visualization of word clouds for each cluster using Plotly
    to maintain consistency with create_cluster_word_viz()
    """
    n_clusters = len(clusters)
    n_cols = min(3, n_clusters)
    n_rows = math.ceil(n_clusters / n_cols)
    
    fig = make_subplots(
        rows=n_rows, 
        cols=n_cols,
        subplot_titles=[f'Cluster {i+1}' for i in range(n_clusters)],
        vertical_spacing=0.15,
        horizontal_spacing=0.05
    )
    # Increase font size of subplot titles
    for annotation in fig.layout.annotations:
        annotation.font.size = 30  # Font size of subplot titles

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def get_text_dimensions(term, size):
        """Estimate text dimensions based on term length and font size"""
        char_width = size / 2  # Approximate width per character
        char_height = size  # Approximate height
        return len(term) * char_width / 100, char_height / 100  # Scale down to plot units
    
    def check_overlap(x1, y1, w1, h1, x2, y2, w2, h2, padding=0.1):
        """Check if two text boxes overlap"""
        return not (x1 + w1/2 + padding < x2 - w2/2 or
                   x1 - w1/2 - padding > x2 + w2/2 or
                   y1 + h1/2 + padding < y2 - h2/2 or
                   y1 - h1/2 - padding > y2 + h2/2)
    
    def generate_positions(terms, sizes):
        """Generate non-overlapping positions for terms"""
        positions = []
        
        # Start with center position for first term
        first_width, first_height = get_text_dimensions(terms[0], sizes[0])
        positions.append((0, 0, first_width, first_height))
        
        # Place remaining terms
        for i in range(1, len(terms)):
            term = terms[i]
            size = sizes[i]
            width, height = get_text_dimensions(term, size)
            
            # Try positions at increasing distances from center
            base_radius = 0.3  # Start closer to center
            radius_step = 0.1
            angle_step = np.pi / 8
            
            for radius in np.arange(base_radius, 1.0, radius_step):
                for angle in np.arange(0, 2*np.pi, angle_step):
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)
                    
                    # Check overlap with all existing terms
                    overlaps = False
                    for px, py, pw, ph in positions:
                        if check_overlap(x, y, width, height, px, py, pw, ph):
                            overlaps = True
                            break
                    
                    if not overlaps:
                        positions.append((x, y, width, height))
                        break
                
                if len(positions) > i:  # If position was found
                    break
            
            # If no position found, place at default position
            if len(positions) <= i:
                positions.append((radius * np.cos(i * 2*np.pi/len(terms)), 
                                radius * np.sin(i * 2*np.pi/len(terms)),
                                width, height))
        
        return [(x, y) for x, y, w, h in positions]
    
    for idx, (_, info) in enumerate(clusters.items()):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        color = colors[idx % len(colors)]
        
        terms = info['top_terms'][:max_terms]
        
        # Calculate term sizes
        base_size = 50
        size_decay = 0.85
        term_sizes = [base_size * (size_decay ** i) for i in range(len(terms))]
        
        # Generate non-overlapping positions
        positions = generate_positions(terms, term_sizes)
        
        # Add terms to plot
        for i, (term, size, (x, y)) in enumerate(zip(terms, term_sizes, positions)):
            opacity = 1.0 - (i * 0.5 / len(terms))
            
            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    text=[term],
                    mode='text',
                    textfont=dict(
                        size=size,
                        color=color,
                        family='Arial Black' if i == 0 else 'Arial'
                    ),
                    opacity=opacity,
                    hoverinfo='text',
                    hovertext=f'Term: {term}',
                    showlegend=False
                ),
                row=row,
                col=col
            )
            
    fig.update_layout(
        title_text="Topic Clusters - Word Importance",
        title_x=0.5,
        title_font_size=35,
        height=600 * n_rows,
        width=1500,
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    for i in range(1, n_clusters + 1):
        fig.update_xaxes(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False, 
            range=[-1.2, 1.2], 
            row=(i-1)//n_cols + 1, 
            col=(i-1)%n_cols + 1
        )
        fig.update_yaxes(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False, 
            range=[-1.2, 1.2], 
            row=(i-1)//n_cols + 1, 
            col=(i-1)%n_cols + 1
        )
    
    return fig




if __name__ == "__main__":
    TERMS = 6
    docs = {i: tweet for i, tweet in enumerate(df['tweet'])}
    # docs = dict(list(docs.items())[:100000])  # First 1000 tweets
    
    print("Processing hashtags...")
    docs = dict(zip(docs.keys(), map(process_tweet, docs.values())))
    
    # Find topic clusters
    # TODO: Now it's about tuning these parameters to get the best results
    clusters = find_topic_clusters(
        docs,
        threshold=0.2, #0.25
        num_perm=120,
        q=4, #4
        min_cluster_size=40, # 20 #30
        n_terms=TERMS, #10
        batch_size=1000,
        n_clusters=6,
        max_docs=300_000
    )
    
    print(f"\nFound {len(clusters)} topic clusters:")
    for cluster_id, info in clusters.items():
        print(f"\nCluster {cluster_id}:")
        print(f"Size: {info['size']} documents")
        print(f"Top terms: {', '.join(info['top_terms'])}")
        print("\nExample documents:")
        for doc_id in info['documents'][:3]:
            print(f"- {docs[doc_id][:100]}...")
        print(24*"-")

    fig = create_cluster_wordclouds(clusters, max_terms=TERMS)
    fig.show()