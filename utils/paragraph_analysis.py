@app.route('/paragraph-analysis')
def paragraph_analysis():
    """Paragraph analysis dashboard view."""
    # Get filter parameters
    doc_ids = request.args.getlist('doc_ids', type=int)
    similarity_threshold = request.args.get('similarity', type=float, default=0.7)
    para_type = request.args.get('para_type', default='all')
    
    # If no documents selected, default to all
    if not doc_ids:
        doc_ids = [doc.id for doc in Document.query.all()]
    
    # Initialize the analyzer
    analyzer = DocumentAnalyzer()
    analyzer.load_data()
    
    # Get paragraph statistics
    stats = analyzer.get_paragraph_stats()
    
    # Get paragraph clusters
    clusters = analyzer.get_paragraph_clusters(min_similarity=similarity_threshold)
    
    # Ensure each cluster has a match_type (add if missing)
    for cluster in clusters:
        if 'match_type' not in cluster:
            # Default to 'similar' if not specified
            cluster['match_type'] = 'similar'
    
    # Filter clusters by paragraph type if specified
    if para_type != 'all':
        clusters = [c for c in clusters if c.get('paragraph_type') == para_type]
    
    # Filter clusters to only include selected documents
    filtered_clusters = []
    for cluster in clusters:
        # Check if any paragraphs in the cluster are from selected documents
        paragraphs_in_selected_docs = [p for p in cluster['paragraphs'] 
                                      if p.document_id in doc_ids]
        
        # Only include if at least 2 paragraphs from selected documents
        if len(paragraphs_in_selected_docs) >= 2:
            # Create a copy of the cluster with only the selected paragraphs
            filtered_cluster = cluster.copy()
            filtered_cluster['paragraphs'] = paragraphs_in_selected_docs
            filtered_cluster['paragraph_count'] = len(paragraphs_in_selected_docs)
            filtered_cluster['document_count'] = len(set(p.document_id for p in paragraphs_in_selected_docs))
            
            # Ensure match_type is present
            if 'match_type' not in filtered_cluster:
                filtered_cluster['match_type'] = 'similar'
                
            filtered_clusters.append(filtered_cluster)
    
    # Generate visualizations
    document_network = analyzer.generate_document_network_viz(threshold=0.3)
    paragraph_length_chart = analyzer.generate_paragraph_length_distribution()
    
    # Generate heatmap only if 2 or more documents are selected 
    paragraph_heatmap = ''
    if len(doc_ids) > 1:
        paragraph_heatmap = analyzer.generate_paragraph_heatmap(doc_ids)
    
    # Generate paragraph type distribution chart
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO
    
    # Get paragraph type counts
    para_type_counts = stats['paragraphs_by_type']
    
    plt.figure(figsize=(8, 6))
    labels = list(para_type_counts.keys())
    values = list(para_type_counts.values())
    colors = ['#dc3545', '#198754', '#fd7e14', '#6610f2', '#20c997']
    
    plt.pie(values, labels=labels, autopct='%1.1f%%', colors=colors[:len(labels)], 
            startangle=90, shadow=False)
    plt.axis('equal')
    plt.title('Paragraph Types Distribution')
    
    # Save chart to base64
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    paragraph_type_chart = f'<img src="data:image/png;base64,{base64.b64encode(buf.read()).decode("utf-8")}" class="img-fluid" />'
    
    # Calculate metrics
    metrics = {
        'total_paragraphs': sum(len(Paragraph.query.filter_by(document_id=doc_id).all()) for doc_id in doc_ids),
        'duplicate_paragraphs': sum(c['paragraph_count'] for c in filtered_clusters if c.get('match_type') == 'exact'),
        'similar_paragraphs': sum(c['paragraph_count'] for c in filtered_clusters if c.get('match_type') == 'similar'),
        'avg_similarity': sum(c.get('similarity', 0.9) for c in filtered_clusters) / len(filtered_clusters) if filtered_clusters else 0
    }
    
    # Get unique paragraph types for tabs
    paragraph_types = set(c.get('paragraph_type', 'regular') for c in filtered_clusters)
    
    # Get all documents for filter dropdown
    all_documents = Document.query.all()
    
    return render_template('paragraph_analysis.html',
                      all_documents=all_documents,
                      selected_doc_ids=doc_ids,
                      similarity_threshold=similarity_threshold,
                      para_type=para_type,
                      metrics=metrics,
                      paragraph_clusters=filtered_clusters,
                      filtered_clusters=filtered_clusters,  # For the template
                      paragraph_types=paragraph_types,
                      document_network=document_network,
                      paragraph_type_chart=paragraph_type_chart,
                      paragraph_length_chart=paragraph_length_chart,
                      paragraph_heatmap=paragraph_heatmap)
