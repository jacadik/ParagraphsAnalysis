# utils/comparison.py
from difflib import SequenceMatcher
import re
from models import Document, Paragraph

def preprocess_text(text):
    """Preprocess text for better comparison."""
    # Convert to lowercase
    text = text.lower()
    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove common punctuation
    text = re.sub(r'[,.;:!?"\'\(\)\[\]]', '', text)
    return text

def calculate_similarity(text1, text2):
    """Calculate the similarity between two text strings."""
    # Preprocess texts
    processed_text1 = preprocess_text(text1)
    processed_text2 = preprocess_text(text2)
    
    # Use SequenceMatcher for similarity calculation
    matcher = SequenceMatcher(None, processed_text1, processed_text2)
    similarity = matcher.ratio()
    
    return similarity

def get_paragraph_matches(doc1_id, doc2_id, threshold=0.7):
    """Find matching paragraphs between two documents."""
    # Get paragraphs from both documents
    doc1_paragraphs = Paragraph.query.filter_by(document_id=doc1_id).order_by(Paragraph.index).all()
    doc2_paragraphs = Paragraph.query.filter_by(document_id=doc2_id).order_by(Paragraph.index).all()
    
    matches = []
    
    # First, try matching by similarity hash (exact matches)
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
    
    # For paragraphs without exact matches, calculate similarity
    for p1 in doc1_paragraphs:
        if p1.id in hash_matches:
            # Already has exact matches
            matches.append({
                'paragraph': p1,
                'matches': hash_matches[p1.id]
            })
            continue
            
        para_matches = []
        
        for p2 in doc2_paragraphs:
            # Skip if this paragraph already has an exact match
            if any(m.get('paragraph').id == p2.id and m.get('match_type') == 'exact' 
                   for matches_list in matches for m in matches_list.get('matches', [])):
                continue
                
            similarity = calculate_similarity(p1.text, p2.text)
            
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
            
            matches.append({
                'paragraph': p1,
                'matches': para_matches
            })
    
    return matches

def compare_documents(doc1_id, doc2_id):
    """Compare two documents and return comparison data."""
    # Get document info
    doc1 = Document.query.get(doc1_id)
    doc2 = Document.query.get(doc2_id)
    
    # Get paragraph matches
    matches = get_paragraph_matches(doc1_id, doc2_id)
    
    # Generate comparison summary
    exact_match_count = sum(1 for m in matches 
                           for match in m['matches'] if match['match_type'] == 'exact')
    similar_match_count = sum(1 for m in matches 
                             for match in m['matches'] if match['match_type'] == 'similar')
    
    doc1_total_paragraphs = Paragraph.query.filter_by(document_id=doc1_id).count()
    doc2_total_paragraphs = Paragraph.query.filter_by(document_id=doc2_id).count()
    
    # Paragraphs in doc1 without matches
    unmapped_paragraphs = [p for p in Paragraph.query.filter_by(document_id=doc1_id).all() 
                          if not any(m['paragraph'].id == p.id for m in matches)]
    
    # Calculate overall similarity
    if doc1_total_paragraphs > 0:
        match_percentage = (exact_match_count + similar_match_count) / doc1_total_paragraphs * 100
    else:
        match_percentage = 0
    
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
            'match_percentage': match_percentage
        }
    }

def find_similar_documents(doc_id, threshold=0.6, limit=5):
    """Find documents similar to the given document."""
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
        matches = get_paragraph_matches(doc_id, other_doc.id, threshold=threshold)
        
        # Calculate a similarity score
        source_paragraphs = Paragraph.query.filter_by(document_id=doc_id).count()
        matched_paragraphs = len(matches)
        
        if source_paragraphs > 0:
            similarity = matched_paragraphs / source_paragraphs
        else:
            similarity = 0
            
        similarity_scores.append({
            'document': other_doc,
            'similarity': similarity,
            'matched_paragraphs': matched_paragraphs,
            'total_paragraphs': source_paragraphs
        })
    
    # Sort by similarity (highest first)
    similarity_scores.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Limit results
    return similarity_scores[:limit]

def highlight_differences(text1, text2):
    """Generate HTML with highlighted differences between two texts."""
    import difflib
    
    # Create a diff
    diff = difflib.HtmlDiff()
    
    # Split into lines
    text1_lines = text1.splitlines()
    text2_lines = text2.splitlines()
    
    # Generate HTML
    html = diff.make_table(text1_lines, text2_lines, 
                          'Document 1', 'Document 2',
                          context=True, numlines=2)
    
    return html
