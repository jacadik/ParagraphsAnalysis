# app.py
import os
from flask import Flask, request, render_template, redirect, url_for, flash, send_file, jsonify
from werkzeug.utils import secure_filename
from models import db, Document, Paragraph, Tag
from utils.document_processor import save_uploaded_file, process_document
from utils.export import generate_document_excel

def create_app():
    # App configuration
    app = Flask(__name__)
    app.config.from_object('config')

    # Initialize database
    db.init_app(app)

    # Ensure directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('static/exports', exist_ok=True)

    # Import comparison utilities
    from utils.comparison import compare_documents, find_similar_documents, highlight_differences, find_common_paragraphs, get_document_clustering
    from utils.analyzer import DocumentAnalyzer
    from utils.ocr_processor import OCRProcessor

    # Create tables - this function is used with app context
    with app.app_context():
        db.create_all()

    # Helper functions
    def allowed_file(filename):
        """Check if the file has an allowed extension."""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

    # Routes
    @app.route('/')
    def index():
        """Home page with upload form."""
        return render_template('index.html')

    @app.route('/upload', methods=['POST'])
    def upload_file():
        """Handle document upload."""
        if 'documents' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        
        files = request.files.getlist('documents')
        
        if not files or files[0].filename == '':
            flash('No files selected', 'danger')
            return redirect(request.url)
        
        # Process each file
        processed_count = 0
        for file in files:
            if file and allowed_file(file.filename):
                try:
                    file_info = save_uploaded_file(file, app.config['UPLOAD_FOLDER'])
                    process_document(file_info)
                    processed_count += 1
                except Exception as e:
                    flash(f'Error processing {file.filename}: {str(e)}', 'danger')
        
        if processed_count > 0:
            flash(f'Successfully processed {processed_count} documents', 'success')
        
        return redirect(url_for('document_list'))

    @app.route('/documents')
    def document_list():
        """List all uploaded documents."""
        documents = Document.query.order_by(Document.upload_date.desc()).all()
        return render_template('document_list.html', documents=documents)

    @app.route('/document/<int:id>')
    def document_view(id):
        """View document details including paragraphs."""
        document = Document.query.get_or_404(id)
        all_tags = Tag.query.all()
        return render_template('document_view.html', document=document, all_tags=all_tags)

    @app.route('/document/<int:id>/download')
    def document_download(id):
        """Download original document."""
        document = Document.query.get_or_404(id)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], document.filename)
        return send_file(file_path, as_attachment=True, download_name=document.original_filename)

    @app.route('/document/<int:id>/delete', methods=['POST'])
    def delete_document(id):
        """Delete a document and its associated paragraphs."""
        document = Document.query.get_or_404(id)
        
        try:
            # Delete the physical file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], document.filename)
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # Delete from database (paragraphs will be deleted via cascade)
            db.session.delete(document)
            db.session.commit()
            
            flash(f'Document "{document.original_filename}" deleted successfully', 'success')
        except Exception as e:
            db.session.rollback()
            flash(f'Error deleting document: {str(e)}', 'danger')
        
        return redirect(url_for('document_list'))

    @app.route('/documents/delete-all', methods=['POST'])
    def delete_all_documents():
        """Delete all documents and associated paragraphs."""
        try:
            # Get all documents
            documents = Document.query.all()
            
            # Delete physical files
            for document in documents:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], document.filename)
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            # Use SQLAlchemy to delete all documents (cascade will handle paragraphs)
            Document.query.delete()
            db.session.commit()
            
            flash('All documents deleted successfully', 'success')
        except Exception as e:
            db.session.rollback()
            flash(f'Error deleting documents: {str(e)}', 'danger')
        
        return redirect(url_for('document_list'))

    @app.route('/documents/delete-selected', methods=['POST'])
    def delete_selected_documents():
        """Delete selected documents."""
        selected_ids = request.form.getlist('selected_docs')
        
        if not selected_ids:
            flash('No documents selected', 'warning')
            return redirect(url_for('document_list'))
        
        try:
            # Convert string IDs to integers
            doc_ids = [int(id) for id in selected_ids]
            
            # Get the documents
            documents = Document.query.filter(Document.id.in_(doc_ids)).all()
            
            # Delete physical files
            for document in documents:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], document.filename)
                if os.path.exists(file_path):
                    os.remove(file_path)
                db.session.delete(document)
            
            db.session.commit()
            
            flash(f'Successfully deleted {len(documents)} documents', 'success')
        except Exception as e:
            db.session.rollback()
            flash(f'Error deleting documents: {str(e)}', 'danger')
        
        return redirect(url_for('document_list'))

    @app.route('/document/<int:id>/tag', methods=['POST'])
    def tag_document(id):
        """Add tag to document."""
        document = Document.query.get_or_404(id)
        tag_id = request.form.get('tag_id')
        
        if tag_id:
            tag = Tag.query.get_or_404(tag_id)
            if tag not in document.tags:
                document.tags.append(tag)
                db.session.commit()
                flash(f'Added tag "{tag.name}" to document', 'success')
        
        return redirect(url_for('document_view', id=id))

    @app.route('/paragraph/<int:id>/tag', methods=['POST'])
    def tag_paragraph(id):
        """Add tag to paragraph."""
        paragraph = Paragraph.query.get_or_404(id)
        tag_id = request.form.get('tag_id')
        
        if tag_id:
            tag = Tag.query.get_or_404(tag_id)
            if tag not in paragraph.tags:
                paragraph.tags.append(tag)
                db.session.commit()
                flash(f'Added tag "{tag.name}" to paragraph', 'success')
        
        return redirect(url_for('document_view', id=paragraph.document_id))

    @app.route('/tags', methods=['GET', 'POST'])
    def tag_manager():
        """Manage tags."""
        if request.method == 'POST':
            name = request.form.get('name')
            color = request.form.get('color', '#17a2b8')
            
            if name:
                # Check if tag exists
                existing_tag = Tag.query.filter_by(name=name).first()
                if existing_tag:
                    flash(f'Tag "{name}" already exists', 'warning')
                else:
                    tag = Tag(name=name, color=color)
                    db.session.add(tag)
                    db.session.commit()
                    flash(f'Created tag "{name}"', 'success')
            
        tags = Tag.query.all()
        return render_template('tag_manager.html', tags=tags)

    @app.route('/export')
    def export_data():
        """Export data to Excel."""
        documents = Document.query.all()
        excel_path = generate_document_excel(documents)
        return send_file(excel_path, as_attachment=True, download_name='document_report.xlsx')

    @app.route('/compare')
    def compare_selection():
        """Select documents to compare."""
        documents = Document.query.order_by(Document.upload_date.desc()).all()
        return render_template('compare_selection.html', documents=documents)

    @app.route('/compare/results', methods=['GET', 'POST'])
    def compare_results():
        """Show comparison results."""
        if request.method == 'POST':
            doc1_id = request.form.get('doc1_id', type=int)
            doc2_id = request.form.get('doc2_id', type=int)
        else:
            doc1_id = request.args.get('doc1_id', type=int)
            doc2_id = request.args.get('doc2_id', type=int)
        
        if not doc1_id or not doc2_id:
            flash('Please select two documents to compare', 'warning')
            return redirect(url_for('compare_selection'))
            
        if doc1_id == doc2_id:
            flash('Please select two different documents', 'warning')
            return redirect(url_for('compare_selection'))
            
        # Perform comparison
        comparison_data = compare_documents(doc1_id, doc2_id)
        
        return render_template('compare_results.html', data=comparison_data)

    @app.route('/compare/paragraph/<int:p1_id>/<int:p2_id>')
    def compare_paragraphs(p1_id, p2_id):
        """Detailed comparison of two paragraphs."""
        p1 = Paragraph.query.get_or_404(p1_id)
        p2 = Paragraph.query.get_or_404(p2_id)
        
        # Generate highlighted diff
        diff_html = highlight_differences(p1.text, p2.text)
        
        return render_template('compare_paragraphs.html', 
                             paragraph1=p1, 
                             paragraph2=p2, 
                             diff_html=diff_html)

    @app.route('/document/<int:id>/similar')
    def similar_documents(id):
        """Find documents similar to this one."""
        document = Document.query.get_or_404(id)
        similar_docs = find_similar_documents(id)
        
        return render_template('similar_documents.html', 
                             document=document, 
                             similar_docs=similar_docs)
                             
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

    @app.route('/process-ocr/<int:id>', methods=['POST'])
    def process_ocr(id):
        """Process a document with OCR."""
        document = Document.query.get_or_404(id)
        
        try:
            # Delete existing paragraphs
            Paragraph.query.filter_by(document_id=id).delete()
            
            # Load OCR processor
            ocr = OCRProcessor()
            
            # Get file path
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], document.filename)
            
            # Process with OCR
            paragraphs = ocr.process_document(file_path)
            
            # Add paragraphs to database
            from utils.document_processor import compute_paragraph_hash
            
            for idx, (text, para_type) in enumerate(paragraphs):
                if text.strip():  # Skip empty paragraphs
                    similarity_hash = compute_paragraph_hash(text)
                    paragraph = Paragraph(
                        document_id=document.id,
                        text=text,
                        paragraph_type=para_type,
                        index=idx,
                        similarity_hash=similarity_hash
                    )
                    db.session.add(paragraph)
            
            db.session.commit()
            
            flash('Document processed with OCR successfully', 'success')
        except Exception as e:
            db.session.rollback()
            flash(f'Error processing with OCR: {str(e)}', 'danger')
        
        return redirect(url_for('document_view', id=id))

    @app.route('/api/document-clusters')
    def document_clusters_api():
        """API endpoint to get document clusters."""
        clusters = get_document_clustering()
        
        # Format for API response
        result = []
        for cluster in clusters:
            doc_list = []
            for doc in cluster['documents']:
                doc_list.append({
                    'id': doc.id,
                    'name': doc.original_filename,
                    'type': doc.file_type,
                    'upload_date': doc.upload_date.strftime('%Y-%m-%d %H:%M')
                })
            
            result.append({
                'size': cluster['size'],
                'documents': doc_list,
                'center_document': {
                    'id': cluster['center_document'].id,
                    'name': cluster['center_document'].original_filename
                }
            })
        
        return jsonify(result)

    @app.route('/api/common-paragraphs')
    def common_paragraphs_api():
        """API endpoint to get common paragraphs across documents."""
        doc_ids = request.args.getlist('doc_ids', type=int)
        
        if not doc_ids or len(doc_ids) < 2:
            return jsonify({'error': 'Please select at least two documents'}), 400
        
        common_paragraphs = find_common_paragraphs(doc_ids)
        
        # Format for API response
        result = []
        for group in common_paragraphs:
            paragraphs = []
            for para in group['paragraphs']:
                doc = Document.query.get(para.document_id)
                paragraphs.append({
                    'id': para.id,
                    'text': para.text,
                    'document_id': para.document_id,
                    'document_name': doc.original_filename if doc else 'Unknown',
                    'paragraph_type': para.paragraph_type,
                    'index': para.index
                })
            
            result.append({
                'match_type': group.get('match_type', 'similar'),  # Ensure match_type is present
                'document_count': group['document_count'],
                'paragraph_count': group['paragraph_count'],
                'paragraphs': paragraphs,
                'sample_text': group['sample_text']
            })
        
        return jsonify(result)

    return app

# Create the application instance
app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
