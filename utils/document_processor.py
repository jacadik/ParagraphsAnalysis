# utils/document_processor.py
import os
import fitz  # PyMuPDF
import docx
import re
from datetime import datetime
from werkzeug.utils import secure_filename
from models import db, Document, Paragraph

def save_uploaded_file(file, upload_folder):
    """Save the uploaded file to the uploads folder."""
    filename = secure_filename(file.filename)
    # Add timestamp to filename to avoid duplicates
    base, ext = os.path.splitext(filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_filename = f"{base}_{timestamp}{ext}"
    
    file_path = os.path.join(upload_folder, unique_filename)
    file.save(file_path)
    
    return {
        'original_filename': filename,
        'filename': unique_filename,
        'file_path': file_path,
        'file_type': ext[1:].lower()  # Remove the dot from extension
    }

def extract_text_from_pdf(file_path):
    """Extract text from PDF files using PyMuPDF with enhanced paragraph detection."""
    doc = fitz.open(file_path)
    paragraphs = []
    paragraph_types = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Get text blocks with their bounding boxes and other properties
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            # Skip image blocks
            if block["type"] == 1:  # Image blocks are type 1
                continue
                
            for line in block["lines"]:
                # Analyze text formatting
                spans = line["spans"]
                
                # Skip empty spans
                if not spans:
                    continue
                
                # Get text from all spans in this line
                line_text = " ".join(span["text"] for span in spans if span["text"].strip())
                
                if not line_text.strip():
                    continue
                
                # Analyze formatting to determine paragraph type
                is_heading = False
                is_list_item = False
                
                # Check for heading (larger font or bold)
                if any(span.get("flags", 0) & 16 for span in spans):  # Check for bold text
                    is_heading = True
                elif any(span.get("size", 0) > 12 for span in spans):  # Larger font could be a heading
                    is_heading = True
                
                # Check for list item
                if line_text.strip().startswith(("•", "-", "*", "○", "▪", "■")) or re.match(r"^\d+\.\s", line_text.strip()):
                    is_list_item = True
                
                # Determine paragraph type
                if is_heading:
                    para_type = "heading"
                elif is_list_item:
                    para_type = "list-item"
                else:
                    para_type = "regular"
                
                # Add to paragraphs
                paragraphs.append(line_text)
                paragraph_types.append(para_type)
    
    # Merge regular paragraphs that should be together
    merged_paragraphs = []
    merged_types = []
    current_paragraph = ""
    current_type = ""
    
    for i, (text, para_type) in enumerate(zip(paragraphs, paragraph_types)):
        if not current_paragraph:
            current_paragraph = text
            current_type = para_type
        elif para_type == "regular" and current_type == "regular" and not text.endswith((".", "!", "?")):
            # Continue current paragraph
            current_paragraph += " " + text
        else:
            # Save current and start new
            if current_paragraph:
                merged_paragraphs.append(current_paragraph)
                merged_types.append(current_type)
            current_paragraph = text
            current_type = para_type
    
    # Add last paragraph if any
    if current_paragraph:
        merged_paragraphs.append(current_paragraph)
        merged_types.append(current_type)
    
    # Return paragraphs with their types
    return list(zip(merged_paragraphs, merged_types))

def extract_text_from_docx(file_path):
    """Extract text from DOCX files using python-docx with enhanced paragraph detection."""
    doc = docx.Document(file_path)
    result = []
    
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
            
        # Determine paragraph type based on style and content
        style_name = paragraph.style.name.lower()
        is_heading = "heading" in style_name or style_name.startswith("h")
        is_list_item = paragraph.style.name.lower() in ("list paragraph", "list bullet", "list number") or \
                      text.startswith(("•", "-", "*")) or re.match(r"^\d+\.\s", text)
        
        if is_heading:
            para_type = "heading"
        elif is_list_item:
            para_type = "list-item"
        else:
            para_type = "regular"
            
        # Check if paragraph has any runs with bold formatting
        if any(run.bold for run in paragraph.runs if hasattr(run, 'bold') and run.bold):
            para_type = "heading"
            
        result.append((text, para_type))
    
    # Also check for tables
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    text = paragraph.text.strip()
                    if text:
                        result.append((f"[TABLE] {text}", "table-cell"))
    
    return result

def compute_paragraph_hash(text):
    """Compute a similarity hash for a paragraph to help with comparison."""
    import hashlib
    import re
    
    # Normalize text: lowercase, remove punctuation, normalize whitespace
    normalized = re.sub(r'[^\w\s]', '', text.lower())
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    # Compute hash
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()

def process_document(file_info):
    """Process document and store text in database with enhanced paragraph detection."""
    file_path = file_info['file_path']
    file_type = file_info['file_type']
    
    # Create document record
    document = Document(
        filename=file_info['filename'],
        original_filename=file_info['original_filename'],
        file_type=file_type
    )
    db.session.add(document)
    db.session.flush()  # Get ID without committing transaction
    
    # Extract text based on file type
    if file_type == 'pdf':
        paragraph_data = extract_text_from_pdf(file_path)
    elif file_type in ['docx', 'doc']:
        paragraph_data = extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    # Add paragraphs to database
    for idx, (text, para_type) in enumerate(paragraph_data):
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
    return document.id
