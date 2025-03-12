# utils/export.py
import os
import pandas as pd
from models import Document, Paragraph, Tag

def generate_document_excel(documents):
    """Generate Excel report with document information."""
    data = []
    
    for doc in documents:
        doc_tags = ", ".join([tag.name for tag in doc.tags])
        
        # Add one row per paragraph
        for para in doc.paragraphs:
            para_tags = ", ".join([tag.name for tag in para.tags])
            
            data.append({
                'Document ID': doc.id,
                'Document Name': doc.original_filename,
                'Upload Date': doc.upload_date.strftime('%Y-%m-%d %H:%M'),
                'Document Tags': doc_tags,
                'Paragraph ID': para.id,
                'Paragraph Index': para.index + 1,  # 1-based for users
                'Paragraph Type': para.paragraph_type,
                'Paragraph Text': para.text,
                'Paragraph Tags': para_tags,
                'Similarity Hash': para.similarity_hash
            })
    
    # Create DataFrame and export to Excel
    df = pd.DataFrame(data)
    
    # Ensure export directory exists
    export_dir = 'static/exports'
    os.makedirs(export_dir, exist_ok=True)
    
    excel_path = f'{export_dir}/document_report.xlsx'
    
    # Write Excel file
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Documents', index=False)
        
        # Auto-adjust columns
        worksheet = writer.sheets['Documents']
        for idx, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.set_column(idx, idx, min(max_len, 100))  # Cap at 100 to avoid very wide columns
    
    return excel_path
