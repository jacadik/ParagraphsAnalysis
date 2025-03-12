# utils/comparison.py
from difflib import SequenceMatcher
import re
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from models import Document, Paragraph, db

def preprocess_text(text):
    """Preprocess text for better comparison."""
    # Convert to lowercase
    text = text.lower()
    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove common punctuation
    text = re.sub(r'[,.;:!?"\'\(\)\[\]]', '', text)
    return text

def calculate_similarity(text1, text2, method="hybrid"):
    """
    Calculate the similarity between two text strings.
    
    Args:
        text1: First text string
        text2: Second text string
        method: Similarity method ('sequence', 'cosine', or 'hybrid')
        
    Returns:
        Similarity score between 0 and 1
    """
    # Preprocess texts
    processed_text1 = preprocess_text(text1)
    processed_text2 = preprocess_text(text2)
    
    if method == "sequence" or (len(processed_text1) < 100 and len(processed_text2) < 100):
        # Use SequenceMatcher for short texts or if explicitly requested
        matcher = SequenceMatcher(None, processed_text1, processed_text2)
        return matcher.ratio()
    
    elif method == "cosine":
        # Use TF-IDF and cosine similarity for longer texts
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([processed_text1, processed_text2])
            return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            # Fall back to sequence matcher if vectorizer fails
            matcher = SequenceMatcher(None, processed_text1, processed_text2)
            return matcher.ratio()
    
    else:  # hybrid - default
        # Short texts: use sequence matcher
        if len(processed_text1) < 100 and len(processed_text2) < 100:
            matcher = SequenceMatcher(None, processed_text1, processed_text2)
            return matcher.ratio()
        
        # Longer texts: use a combination
        try:
            # Calculate word overlap (Jaccard similarity)
            words1 = set(processed_text1.split())
            words2 = set(processed_text2.split())
            jaccard = len(words1.intersection(words2)) / len(words1.union(words2)) if words1 or words2 else 0
            
            # Calculate TF-IDF cosine similarity
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([processed_text1, processed_text2])
            cosine = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Weighted average (TF-IDF is better for longer texts)
            return 0.3 * jaccard + 0.7 * cosine
        except:
            # Fall back to sequence matcher if advanced methods fail
            matcher = SequenceMatcher(None, processed_text1, processed_text2)
            return matcher.ratio()

def get_paragraph_matches(doc1_id, doc2_id, threshold=0.7, method="hybrid"):
    """
    Find matching paragraphs between two documents with enhanced similarity detection.
    
    Args:
        doc1_id: ID of the first document
        doc2_id: ID of the second document
        threshold: Minimum similarity threshold (0.0-1.0)
        method: Similarity calculation method
        
    Returns:
        List of matching paragraph details
    """
    # Get paragraphs from both documents
    doc1_paragraphs = Paragraph.query.filter_by(document_id=doc1_id).order_by(Paragraph.index).all()
    doc2_paragraphs = Paragraph.query.filter_by(document_id=doc2_id).order_by(Paragraph.index).all()
    
    matches = []
    
    # First, try matching by similarity hash (exact content matches)
    hash_matches = {}
    
    for p1 in doc1_paragraphs:
        for p2 in doc2_paragraphs:
            if p1.similarity_hash == p2.similarity_hash and p1.similarity_hash is not None:
                if p1.id not in hash_matches:
                    hash_matches[p1.id] = []
                hash_matches[p1.id].append({
                    'paragraph': p2,
                    'similarity': 1.0,
                    'match_type': 'exact'
                })
    
    # For paragraphs that are similar but not identical, only compare within same paragraph type
    # This reduces unnecessary comparisons and improves accuracy
    for p1 in doc1_paragraphs:
        if p1.id in hash_matches:
            # Already has exact matches
            matches.append({
                'paragraph': p1,
                'matches': hash_matches[p1.id]
            })
            continue
            
        para_matches = []
        
        # Filter paragraphs by type to reduce comparison space
        # For headings, only compare with other headings, etc.
        potential_matches = [p2 for p2 in doc2_paragraphs if p2.paragraph_type == p1.paragraph_type]
        
        # Skip very short paragraphs (less than 20 chars) as they often lead to false positives
        if len(p1.text.strip()) < 20 and p1.paragraph_type == "regular":
            continue
        
        for p2 in potential_matches:
            # Skip if this paragraph already has an exact match
            if any(m.get('paragraph').id == p2.id and m.get('match_type') == 'exact' 
                   for matches_list in matches for m in matches_list.get('matches', [])):
                continue
                
            # Skip if lengths are too different (quick filter)
            len_ratio = min(len(p1.text), len(p2.text)) / max(len(p1.text), len(p2.text))
            if len_ratio < 0.5:  # If one paragraph is less than half the length of the other
                continue
            
            # Calculate similarity
            similarity = calculate_similarity(p1.text, p2.text, method=method)
            
            if similarity >= threshold:
                para_matches.append({
                    'paragraph': p2,
                    'similarity': similarity,
                    'match_type': 'similar'
                })
        
        # Only include if there are matches
        if para_matches:
            # Sort matches by similarity (highest first)
            para_matches.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Limit to top matches (avoid overwhelming users with too many matches)
            top_matches = para_matches[:5]
            
            matches.append({
                'paragraph': p1,
                'matches': top_matches
            })
    
    return matches

def compare_documents(doc1_id, doc2_id, similarity_method="hybrid"):
    """
    Compare two documents and return detailed comparison data.
    
    Args:
        doc1_id: ID of the first document
        doc2_id: ID of the second document
        similarity_method: Method for calculating similarity
    
    Returns:
        Dictionary with comparison details
    """
    # Get document info
    doc1 = Document.query.get(doc1_id)
    doc2 = Document.query.get(doc2_id)
    
    # Get paragraph matches
    matches = get_paragraph_matches(doc1_id, doc2_id, method=similarity_method)
    
    # Generate comparison summary
    exact_match_count = sum(1 for m in matches 
                           for match in m['matches'] if match['match_type'] == 'exact')
    similar_match_count = sum(1 for m in matches 
                             for match in m['matches'] if match['match_type'] == 'similar')
    
    doc1_total_paragraphs = Paragraph.query.filter_by(document_id=doc1_id).count()
    doc2_total_paragraphs = Paragraph.query.filter_by(document_id=doc2_id).count()
    
    # Paragraphs in doc1 without matches
    all_doc1_paragraphs = {p.id: p for p in Paragraph.query.filter_by(document_id=doc1_id).all()}
    matched_paragraph_ids = {m['paragraph'].id for m in matches}
    unmapped_paragraphs = [p for pid, p in all_doc1_paragraphs.items() if pid not in matched_paragraph_ids]
    
    # Calculate type-specific similarities
    paragraph_type_counts = {'heading': 0, 'list-item': 0, 'regular': 0, 'table-cell': 0, 'table-header': 0}
    paragraph_type_matches = {'heading': 0, 'list-item': 0, 'regular': 0, 'table-cell': 0, 'table-header': 0}
    
    # Count paragraphs by type
    for p in all_doc1_paragraphs.values():
        if p.paragraph_type in paragraph_type_counts:
            paragraph_type_counts[p.paragraph_type] += 1
    
    # Count matches by type
    for m in matches:
        para_type = m['paragraph'].paragraph_type
        if para_type in paragraph_type_matches:
            paragraph_type_matches[para_type] += 1
    
    # Calculate percentages
    type_percentages = {}
    for ptype in paragraph_type_counts:
        if paragraph_type_counts[ptype] > 0:
            type_percentages[ptype] = paragraph_type_matches[ptype] / paragraph_type_counts[ptype] * 100
        else:
            type_percentages[ptype] = 0
            
    # Calculate overall similarity
    if doc1_total_paragraphs > 0:
        match_percentage = (exact_match_count + similar_match_count) / doc1_total_paragraphs * 100
        weighted_match_percentage = 0
        total_weight = 0
        
        # Weighted similarity - headings are weighted more than regular paragraphs
        weights = {'heading': 3, 'regular': 1, 'list-item': 1.5, 'table-cell': 1, 'table-header': 2}
        
        for ptype, count in paragraph_type_counts.items():
            if count > 0:
                type_weight = weights.get(ptype, 1)
                weighted_match_percentage += (paragraph_type_matches[ptype] / count * 100) * type_weight
                total_weight += type_weight
        
        if total_weight > 0:
            weighted_match_percentage /= total_weight
    else:
        match_percentage = 0
        weighted_match_percentage = 0
    
    return {
        'doc1': doc1,
        'doc2': doc2,
        'matches': matches,
        'unmapped_paragraphs': unmapped_paragraphs,
        'summary': {
            'doc1_paragraphs': doc1_total_paragraphs,
            'doc2_paragraphs': doc2_total_paragraphs,
            'exact_matches': exact_match_count,
            'similar_matches': similar_match_count,
            'match_percentage': match_percentage,
            'weighted_match_percentage': weighted_match_percentage,
            'type_matches': paragraph_type_matches,
            'type_totals': paragraph_type_counts,
            'type_percentages': type_percentages
        }
    }

def find_similar_documents(doc_id, threshold=0.6, limit=5, similarity_method="hybrid"):
    """
    Find documents similar to the given document.
    
    Args:
        doc_id: ID of the source document
        threshold: Minimum similarity threshold
        limit: Maximum number of results to return
        similarity_method: Method for calculating similarity
        
    Returns:
        List of similar documents with similarity scores
    """
    # Get the source document
    source_doc = Document.query.get(doc_id)
    if not source_doc:
        return []
        
    # Get all other documents
    other_docs = Document.query.filter(Document.id != doc_id).all()
    
    # Calculate similarity with each document
    similarity_scores = []
    
    for other_doc in other_docs:
        # Get paragraph matches
        matches = get_paragraph_matches(doc_id, other_doc.id, threshold=threshold, method=similarity_method)
        
        # Calculate various similarity metrics
        source_paragraphs = Paragraph.query.filter_by(document_id=doc_id).all()
        source_paragraph_count = len(source_paragraphs)
        
        if source_paragraph_count == 0:
            continue
        
        # Count matches by type
        exact_matches = sum(1 for m in matches for match in m['matches'] if match['match_type'] == 'exact')
        similar_matches = sum(1 for m in matches for match in m['matches'] if match['match_type'] == 'similar')
        matched_paragraphs = len(matches)
        
        # Basic similarity - percentage of paragraphs with matches
        basic_similarity = matched_paragraphs / source_paragraph_count
        
        # Weighted similarity - exact matches count more than similar matches
        weighted_similarity = (exact_matches + 0.7 * similar_matches) / source_paragraph_count
        
        # Detailed paragraph type matching
        paragraph_types = ['heading', 'regular', 'list-item', 'table-cell', 'table-header']
        type_matches = {ptype: 0 for ptype in paragraph_types}
        type_totals = {ptype: 0 for ptype in paragraph_types}
        
        # Count paragraphs by type
        for p in source_paragraphs:
            if p.paragraph_type in type_totals:
                type_totals[p.paragraph_type] += 1
        
        # Count matches by type
        for m in matches:
            para_type = m['paragraph'].paragraph_type
            if para_type in type_matches:
                type_matches[para_type] += 1
        
        # Calculate type-specific similarity
        type_similarity = {}
        for ptype in paragraph_types:
            if type_totals[ptype] > 0:
                type_similarity[ptype] = type_matches[ptype] / type_totals[ptype]
            else:
                type_similarity[ptype] = 0
        
        # Overall similarity score (weighted average)
        weights = {'heading': 3, 'regular': 1, 'list-item': 1.5, 'table-cell': 1, 'table-header': 2}
        weighted_score = 0
        total_weight = 0
        
        for ptype, sim in type_similarity.items():
            if type_totals[ptype] > 0:
                weight = weights.get(ptype, 1)
                weighted_score += sim * weight
                total_weight += weight
        
        if total_weight > 0:
            final_similarity = weighted_score / total_weight
        else:
            final_similarity = basic_similarity
        
        similarity_scores.append({
            'document': other_doc,
            'similarity': final_similarity,
            'basic_similarity': basic_similarity,
            'weighted_similarity': weighted_similarity,
            'type_similarity': type_similarity,
            'matched_paragraphs': matched_paragraphs,
            'total_paragraphs': source_paragraph_count,
            'exact_matches': exact_matches,
            'similar_matches': similar_matches
        })
    
    # Sort by similarity (highest first)
    similarity_scores.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Filter by threshold and limit results
    filtered_scores = [s for s in similarity_scores if s['similarity'] >= threshold]
    return filtered_scores[:limit]

def highlight_differences(text1, text2):
    """Generate HTML with highlighted differences between two texts."""
    import difflib
    
    # Create a diff
    diff = difflib.HtmlDiff()
    
    # Split into lines
    text1_lines = text1.splitlines() or ['']
    text2_lines = text2.splitlines() or ['']
    
    # Generate HTML
    html = diff.make_table(text1_lines, text2_lines, 
                          'Document 1', 'Document 2',
                          context=True, numlines=2)
    
    return html

def find_common_paragraphs(doc_ids, similarity_threshold=0.85, min_length=50):
    """
    Find paragraphs that appear in multiple documents.
    
    Args:
        doc_ids: List of document IDs to compare
        similarity_threshold: Minimum similarity for paragraphs to be considered the same
        min_length: Minimum paragraph length to consider
        
    Returns:
        List of common paragraph groups
    """
    if not doc_ids or len(doc_ids) < 2:
        return []
    
    # Get all documents and their paragraphs
    documents = {}
    all_paragraphs = []
    
    for doc_id in doc_ids:
        doc = Document.query.get(doc_id)
        if not doc:
            continue
            
        paragraphs = Paragraph.query.filter_by(document_id=doc_id).all()
        documents[doc_id] = {
            'document': doc,
            'paragraphs': paragraphs
        }
        
        # Add paragraphs to the global list
        for para in paragraphs:
            if len(para.text) >= min_length:  # Filter out short paragraphs
                all_paragraphs.append(para)
    
    # First, group by hash for exact matches
    hash_groups = {}
    processed_ids = set()
    
    for para in all_paragraphs:
        if para.similarity_hash and para.id not in processed_ids:
            if para.similarity_hash not in hash_groups:
                hash_groups[para.similarity_hash] = []
            hash_groups[para.similarity_hash].append(para)
            processed_ids.add(para.id)
    
    # Then for remaining paragraphs, find similar ones
    similarity_groups = []
    
    for para in all_paragraphs:
        if para.id in processed_ids:
            continue
            
        group = [para]
        processed_ids.add(para.id)
        
        for other_para in all_paragraphs:
            if other_para.id in processed_ids or other_para.document_id == para.document_id:
                continue
                
            similarity = calculate_similarity(para.text, other_para.text)
            if similarity >= similarity_threshold:
                group.append(other_para)
                processed_ids.add(other_para.id)
        
        if len(group) > 1:  # Only include groups with paragraphs from multiple documents
            similarity_groups.append(group)
    
    # Combine hash-based and similarity-based groups
    final_groups = []
    
    # Add hash groups that appear in multiple documents
    for hash_value, group in hash_groups.items():
        doc_ids_in_group = set(para.document_id for para in group)
        if len(doc_ids_in_group) > 1:
            final_groups.append({
                'paragraphs': group,
                'match_type': 'exact',
                'document_count': len(doc_ids_in_group),
                'paragraph_count': len(group),
                'sample_text': group[0].text if group else ""
            })
    
    # Add similarity groups
    for group in similarity_groups:
        doc_ids_in_group = set(para.document_id for para in group)
        final_groups.append({
            'paragraphs': group,
            'match_type': 'similar',
            'document_count': len(doc_ids_in_group),
            'paragraph_count': len(group),
            'sample_text': group[0].text if group else ""
        })
    
    # Sort by document count (most common first)
    final_groups.sort(key=lambda x: (x['document_count'], x['paragraph_count']), reverse=True)
    
    return final_groups

def get_document_clustering(similarity_threshold=0.6):
    """
    Cluster documents based on their similarity.
    
    Args:
        similarity_threshold: Minimum similarity for documents to be in the same cluster
        
    Returns:
        List of document clusters
    """
    # Get all documents
    documents = Document.query.all()
    if not documents:
        return []
    
    # Calculate similarity matrix
    doc_count = len(documents)
    similarity_matrix = np.zeros((doc_count, doc_count))
    
    for i in range(doc_count):
        for j in range(i+1, doc_count):
            # Skip if the same document
            if documents[i].id == documents[j].id:
                continue
                
            # Calculate similarity
            matches = get_paragraph_matches(documents[i].id, documents[j].id, threshold=0.7)
            
            if matches:
                doc1_paragraphs = Paragraph.query.filter_by(document_id=documents[i].id).count()
                if doc1_paragraphs > 0:
                    similarity = len(matches) / doc1_paragraphs
                else:
                    similarity = 0
            else:
                similarity = 0
            
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity  # Mirror the matrix
    
    # Perform clustering (simple approach)
    clusters = []
    assigned = set()
    
    for i in range(doc_count):
        if i in assigned:
            continue
            
        cluster = [documents[i]]
        assigned.add(i)
        
        for j in range(doc_count):
            if j in assigned or i == j:
                continue
                
            if similarity_matrix[i, j] >= similarity_threshold:
                cluster.append(documents[j])
                assigned.add(j)
        
        if len(cluster) > 1 or not clusters:  # Only add if multiple docs in cluster or first cluster
            clusters.append({
                'documents': cluster,
                'size': len(cluster),
                'center_document': documents[i]
            })
    
    # Sort clusters by size
    clusters.sort(key=lambda x: x['size'], reverse=True)
    
    return clusters
