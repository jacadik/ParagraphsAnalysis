# app.py
import os
from flask import Flask, request, render_template, redirect, url_for, flash, send_file
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
    from utils.comparison import compare_documents, find_similar_documents, highlight_differences

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

    return app

# Create the application instance
app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
