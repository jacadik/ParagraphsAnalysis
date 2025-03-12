# utils/analyzer.py
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from models import Document, Paragraph, db

class DocumentAnalyzer:
    """A class to analyze document and paragraph relationships."""
    
    def __init__(self):
        """Initialize the analyzer."""
        self.docs = None
        self.paragraphs = None
    
    def load_data(self):
        """Load data from the database."""
        self.docs = Document.query.all()
        self.paragraphs = Paragraph.query.all()
        
        # Create a mapping of document ID to list of paragraph IDs
        self.doc_para_map = defaultdict(list)
        for para in self.paragraphs:
            self.doc_para_map[para.document_id].append(para.id)
    
    def get_paragraph_stats(self):
        """Generate statistics about paragraphs."""
        if not self.paragraphs:
            self.load_data()
        
        stats = {
            'total_paragraphs': len(self.paragraphs),
            'paragraphs_by_type': defaultdict(int),
            'paragraphs_by_doc': defaultdict(int),
            'avg_length': 0,
            'length_distribution': defaultdict(int),
        }
        
        total_length = 0
        for para in self.paragraphs:
            stats['paragraphs_by_type'][para.paragraph_type] += 1
            stats['paragraphs_by_doc'][para.document_id] += 1
            
            para_len = len(para.text)
            total_length += para_len
            
            # Group lengths for distribution (buckets of 50 chars)
            bucket = (para_len // 50) * 50
            stats['length_distribution'][bucket] += 1
        
        if stats['total_paragraphs'] > 0:
            stats['avg_length'] = total_length / stats['total_paragraphs']
        
        return stats
    
    def get_duplicate_paragraphs(self, threshold=0.9):
        """Find duplicate paragraphs across all documents."""
        from utils.comparison import calculate_similarity
        
        if not self.paragraphs:
            self.load_data()
            
        # First check for exact matches using hash
        exact_matches = defaultdict(list)
        for para in self.paragraphs:
            if para.similarity_hash:
                exact_matches[para.similarity_hash].append(para)
        
        # Filter to only include hashes with multiple paragraphs
        duplicate_groups = []
        for hash_value, paragraphs in exact_matches.items():
            if len(paragraphs) > 1:
                # Check if they come from different documents
                doc_ids = set(p.document_id for p in paragraphs)
                if len(doc_ids) > 1:  # Only include if from multiple documents
                    duplicate_groups.append({
                        'paragraphs': paragraphs,
                        'count': len(paragraphs),
                        'document_count': len(doc_ids),
                        'text': paragraphs[0].text[:200] + ('...' if len(paragraphs[0].text) > 200 else ''),
                        'match_type': 'exact'
                    })
        
        # Now check for near-duplicates using similarity comparison
        # This is more computationally intensive, so we'll focus on paragraphs
        # that are similar in length and type
        similar_groups = []
        processed_ids = set()
        
        # Get the paragraphs that aren't in exact match groups
        used_in_exact = set()
        for group in duplicate_groups:
            for para in group['paragraphs']:
                used_in_exact.add(para.id)
        
        remaining_paragraphs = [p for p in self.paragraphs if p.id not in used_in_exact]
        
        # Group paragraphs by type for more efficient comparison
        paragraphs_by_type = defaultdict(list)
        for para in remaining_paragraphs:
            paragraphs_by_type[para.paragraph_type].append(para)
        
        # For each paragraph type, compare paragraphs
        for para_type, type_paragraphs in paragraphs_by_type.items():
            for i, para1 in enumerate(type_paragraphs):
                if para1.id in processed_ids:
                    continue
                    
                # Skip very short paragraphs as they often lead to false positives
                if len(para1.text) < 50:
                    continue
                
                similar_paras = [para1]
                processed_ids.add(para1.id)
                
                for j in range(i+1, len(type_paragraphs)):
                    para2 = type_paragraphs[j]
                    if para2.id in processed_ids:
                        continue
                    
                    # Quick length check to avoid unnecessary comparisons
                    len_ratio = min(len(para1.text), len(para2.text)) / max(len(para1.text), len(para2.text))
                    if len_ratio < 0.7:
                        continue
                    
                    # Calculate similarity
                    similarity = calculate_similarity(para1.text, para2.text, method="hybrid")
                    
                    if similarity >= threshold:
                        similar_paras.append(para2)
                        processed_ids.add(para2.id)
                
                if len(similar_paras) > 1:
                    # Check if they come from different documents
                    doc_ids = set(p.document_id for p in similar_paras)
                    if len(doc_ids) > 1:  # Only include if from multiple documents
                        similar_groups.append({
                            'paragraphs': similar_paras,
                            'count': len(similar_paras),
                            'document_count': len(doc_ids),
                            'text': similar_paras[0].text[:200] + ('...' if len(similar_paras[0].text) > 200 else ''),
                            'match_type': 'similar'
                        })
        
        # Combine and sort the results
        all_groups = duplicate_groups + similar_groups
        all_groups.sort(key=lambda x: (x['document_count'], x['count']), reverse=True)
        
        return all_groups
    
    def get_document_similarity_matrix(self):
        """
        Generate a matrix showing similarity between all documents.
        
        Returns:
            DataFrame: Matrix where both rows and columns are document IDs,
                      and values represent similarity scores.
        """
        from utils.comparison import compare_documents
        
        if not self.docs:
            self.load_data()
            
        # Initialize similarity matrix
        doc_ids = [doc.id for doc in self.docs]
        n_docs = len(doc_ids)
        similarity_matrix = pd.DataFrame(
            np.zeros((n_docs, n_docs)),
            index=doc_ids, 
            columns=doc_ids
        )
        
        # Calculate similarity for each pair of documents
        for i, doc1_id in enumerate(doc_ids):
            similarity_matrix.loc[doc1_id, doc1_id] = 1.0  # Self-similarity is 1.0
            
            for j in range(i+1, n_docs):
                doc2_id = doc_ids[j]
                
                # Compare documents
                comparison = compare_documents(doc1_id, doc2_id)
                similarity = comparison['summary']['weighted_match_percentage'] / 100  # Convert to 0-1 scale
                
                # Store in matrix (symmetric)
                similarity_matrix.loc[doc1_id, doc2_id] = similarity
                similarity_matrix.loc[doc2_id, doc1_id] = similarity
        
        return similarity_matrix
    
    def generate_document_network_viz(self, threshold=0.3):
        """
        Generate a network visualization of document similarities.
        
        Args:
            threshold: Minimum similarity score to include an edge
            
        Returns:
            HTML: Base64 encoded PNG image of the network visualization
        """
        similarity_matrix = self.get_document_similarity_matrix()
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes (documents)
        doc_name_map = {}
        for doc in self.docs:
            doc_name = f"Doc {doc.id}: {doc.original_filename[:20]}"
            if len(doc.original_filename) > 20:
                doc_name += "..."
            doc_name_map[doc.id] = doc_name
            G.add_node(doc.id, label=doc_name, type=doc.file_type)
        
        # Add edges (similarities)
        for i, doc1_id in enumerate(similarity_matrix.index):
            for j, doc2_id in enumerate(similarity_matrix.columns):
                if i < j:  # Only process upper triangle of matrix
                    similarity = similarity_matrix.loc[doc1_id, doc2_id]
                    if similarity >= threshold:
                        G.add_edge(doc1_id, doc2_id, weight=similarity)
        
        # Setup visualization parameters
        plt.figure(figsize=(12, 8))
        
        # Set node colors based on file type
        node_colors = []
        for node in G.nodes():
            if G.nodes[node]['type'] == 'pdf':
                node_colors.append('lightcoral')
            else:  # docx
                node_colors.append('lightskyblue')
        
        # Set edge widths based on similarity
        edge_widths = [G[u][v]['weight'] * 5 for u, v in G.edges()]
        
        # Draw the graph
        pos = nx.spring_layout(G, seed=42)  # Position nodes using force-directed layout
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_colors, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color='gray')
        nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'label'), font_size=8)
        
        # Add edge labels (similarity scores)
        edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        # Add legend
        plt.plot([0], [0], 'o', color='lightcoral', label='PDF Files')
        plt.plot([0], [0], 'o', color='lightskyblue', label='DOCX Files')
        plt.legend(loc='upper left')
        
        plt.title('Document Similarity Network')
        plt.tight_layout()
        plt.axis('off')
        
        # Save the figure to a BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        plt.close()
        
        # Encode the image to base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return f'<img src="data:image/png;base64,{img_str}" alt="Document Network Visualization">'
    
    def generate_paragraph_heatmap(self, doc_ids):
        """
        Generate a heatmap visualization showing paragraph similarities between documents.
        
        Args:
            doc_ids: List of document IDs to include
            
        Returns:
            HTML: Base64 encoded PNG image of the heatmap
        """
        from utils.comparison import calculate_similarity
        
        # Get paragraphs for selected documents
        doc_paragraphs = {}
        for doc_id in doc_ids:
            doc_paragraphs[doc_id] = Paragraph.query.filter_by(document_id=doc_id).order_by(Paragraph.index).all()
        
        # Create labels for rows and columns
        row_labels = []
        para_map = []
        
        for doc_id in doc_ids:
            doc = Document.query.get(doc_id)
            name_prefix = f"Doc {doc_id}: "
            
            for para in doc_paragraphs[doc_id]:
                row_label = f"{name_prefix}Para {para.index+1}"
                if para.paragraph_type == 'heading':
                    # For headings, include the text (truncated if needed)
                    heading_text = para.text[:30]
                    if len(para.text) > 30:
                        heading_text += "..."
                    row_label += f" [{heading_text}]"
                
                row_labels.append(row_label)
                para_map.append((doc_id, para.id))
        
        # Initialize the heatmap matrix
        n_paras = len(para_map)
        heatmap_data = np.zeros((n_paras, n_paras))
        
        # Calculate similarities for the upper triangle of the matrix
        for i in range(n_paras):
            heatmap_data[i, i] = 1.0  # Self-similarity is 1.0
            
            for j in range(i+1, n_paras):
                doc1_id, para1_id = para_map[i]
                doc2_id, para2_id = para_map[j]
                
                # Skip if from same document
                if doc1_id == doc2_id:
                    heatmap_data[i, j] = 0
                    heatmap_data[j, i] = 0
                    continue
                
                # Get paragraph objects
                para1 = Paragraph.query.get(para1_id)
                para2 = Paragraph.query.get(para2_id)
                
                # Calculate similarity only if same paragraph type
                if para1.paragraph_type == para2.paragraph_type:
                    similarity = calculate_similarity(para1.text, para2.text)
                    heatmap_data[i, j] = similarity
                    heatmap_data[j, i] = similarity
        
        # Create heatmap visualization
        plt.figure(figsize=(12, 10))
        plt.imshow(heatmap_data, cmap='YlGnBu', interpolation='nearest')
        plt.colorbar(label='Similarity Score')
        
        # Set tick labels
        plt.xticks(np.arange(n_paras), labels=row_labels, rotation=90, fontsize=8)
        plt.yticks(np.arange(n_paras), labels=row_labels, fontsize=8)
        
        plt.title('Paragraph Similarity Heatmap')
        plt.tight_layout()
        
        # Save the figure to a BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        plt.close()
        
        # Encode the image to base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return f'<img src="data:image/png;base64,{img_str}" alt="Paragraph Similarity Heatmap">'
    
    def get_paragraph_clusters(self, min_similarity=0.85):
        """
        Cluster similar paragraphs across all documents.
        
        Args:
            min_similarity: Minimum similarity for paragraphs to be in the same cluster
            
        Returns:
            List of clusters (paragraphs that are similar to each other)
        """
        from utils.comparison import calculate_similarity
        from sklearn.cluster import DBSCAN
        
        if not self.paragraphs:
            self.load_data()
        
        # Filter out short paragraphs
        filtered_paragraphs = [p for p in self.paragraphs if len(p.text) >= 50]
        
        # Group paragraphs by type to reduce comparison space
        paragraphs_by_type = defaultdict(list)
        for para in filtered_paragraphs:
            paragraphs_by_type[para.paragraph_type].append(para)
        
        all_clusters = []
        
        # Process each paragraph type separately
        for para_type, type_paragraphs in paragraphs_by_type.items():
            if len(type_paragraphs) < 2:
                continue
                
            # Calculate similarity matrix (pairwise distances)
            n_paras = len(type_paragraphs)
            similarity_matrix = np.zeros((n_paras, n_paras))
            
            for i in range(n_paras):
                for j in range(i+1, n_paras):
                    para1 = type_paragraphs[i]
                    para2 = type_paragraphs[j]
                    
                    # Only compare paragraphs from different documents
                    if para1.document_id != para2.document_id:
                        # Quick length filter
                        len_ratio = min(len(para1.text), len(para2.text)) / max(len(para1.text), len(para2.text))
                        if len_ratio >= 0.7:
                            similarity = calculate_similarity(para1.text, para2.text)
                            similarity_matrix[i, j] = similarity
                            similarity_matrix[j, i] = similarity
            
            # Convert similarity to distance (1 - similarity)
            distance_matrix = 1 - similarity_matrix
            
            # Use DBSCAN for clustering
            eps = 1 - min_similarity  # Convert similarity threshold to distance threshold
            clustering = DBSCAN(eps=eps, min_samples=2, metric='precomputed').fit(distance_matrix)
            
            # Extract clusters
            labels = clustering.labels_
            
            # Group paragraphs by cluster
            clusters = defaultdict(list)
            for i, label in enumerate(labels):
                if label != -1:  # Skip noise points
                    clusters[label].append(type_paragraphs[i])
            
            # Add to results
            for cluster_id, paragraphs in clusters.items():
                # Make sure cluster includes paragraphs from different documents
                doc_ids = set(p.document_id for p in paragraphs)
                if len(doc_ids) > 1:
                    # Check if any paragraphs are exact matches by hash
                    has_exact_matches = False
                    if len(paragraphs) >= 2:
                        hash_values = [p.similarity_hash for p in paragraphs if p.similarity_hash]
                        hash_counts = {}
                        for hash_val in hash_values:
                            hash_counts[hash_val] = hash_counts.get(hash_val, 0) + 1
                        has_exact_matches = any(count >= 2 for count in hash_counts.values())
                    
                    all_clusters.append({
                        'paragraphs': paragraphs,
                        'count': len(paragraphs),
                        'document_count': len(doc_ids),
                        'paragraph_type': para_type,
                        'sample_text': paragraphs[0].text[:100] + ('...' if len(paragraphs[0].text) > 100 else ''),
                        'match_type': 'exact' if has_exact_matches else 'similar'  # Add match_type
                    })
        
        # Sort clusters by size (number of paragraphs)
        all_clusters.sort(key=lambda x: (x['document_count'], x['count']), reverse=True)
        
        return all_clusters
    
    def export_data_for_analysis(self):
        """
        Export document and paragraph data for external analysis.
        
        Returns:
            Dictionary with pandas DataFrames
        """
        if not self.docs or not self.paragraphs:
            self.load_data()
        
        # Create document dataframe
        doc_data = []
        for doc in self.docs:
            doc_data.append({
                'document_id': doc.id,
                'filename': doc.original_filename,
                'file_type': doc.file_type,
                'upload_date': doc.upload_date
            })
        doc_df = pd.DataFrame(doc_data)
        
        # Create paragraph dataframe
        para_data = []
        for para in self.paragraphs:
            para_data.append({
                'paragraph_id': para.id,
                'document_id': para.document_id,
                'text': para.text,
                'paragraph_type': para.paragraph_type,
                'index': para.index,
                'similarity_hash': para.similarity_hash,
                'text_length': len(para.text),
                'words': len(para.text.split())
            })
        para_df = pd.DataFrame(para_data)
        
        # Generate summary statistics
        summary = {
            'document_count': len(doc_df),
            'paragraph_count': len(para_df),
            'paragraph_type_counts': para_df['paragraph_type'].value_counts().to_dict(),
            'avg_paragraphs_per_doc': para_df.groupby('document_id').size().mean(),
            'avg_paragraph_length': para_df['text_length'].mean(),
            'avg_words_per_paragraph': para_df['words'].mean()
        }
        
        return {
            'documents': doc_df,
            'paragraphs': para_df,
            'summary': summary
        }
    
    def generate_paragraph_length_distribution(self):
        """
        Generate a visualization of paragraph length distribution.
        
        Returns:
            HTML: Base64 encoded PNG image of the distribution
        """
        if not self.paragraphs:
            self.load_data()
        
        # Get paragraph lengths by type
        para_types = defaultdict(list)
        for para in self.paragraphs:
            para_types[para.paragraph_type].append(len(para.text))
        
        plt.figure(figsize=(12, 8))
        
        # Set color for each paragraph type
        colors = {
            'heading': 'darkorange',
            'regular': 'royalblue',
            'list-item': 'forestgreen',
            'table-cell': 'purple',
            'table-header': 'crimson'
        }
        
        # Create histogram for each paragraph type
        for para_type, lengths in para_types.items():
            if lengths:  # Only plot if there's data
                plt.hist(lengths, bins=30, alpha=0.6, label=para_type, 
                         color=colors.get(para_type, 'gray'))
        
        plt.xlabel('Paragraph Length (characters)')
        plt.ylabel('Count')
        plt.title('Paragraph Length Distribution by Type')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the figure to a BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        plt.close()
        
        # Encode the image to base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return f'<img src="data:image/png;base64,{img_str}" alt="Paragraph Length Distribution">'
