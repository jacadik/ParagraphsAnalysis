# models.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

# Association tables for many-to-many relationships
document_tags = db.Table('document_tags',
    db.Column('document_id', db.Integer, db.ForeignKey('document.id'), primary_key=True),
    db.Column('tag_id', db.Integer, db.ForeignKey('tag.id'), primary_key=True)
)

paragraph_tags = db.Table('paragraph_tags',
    db.Column('paragraph_id', db.Integer, db.ForeignKey('paragraph.id'), primary_key=True),
    db.Column('tag_id', db.Integer, db.ForeignKey('tag.id'), primary_key=True)
)

class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(10), nullable=False)  # 'pdf' or 'docx'
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    paragraphs = db.relationship('Paragraph', backref='document', lazy=True, cascade="all, delete-orphan")
    tags = db.relationship('Tag', secondary=document_tags, lazy='subquery',
                          backref=db.backref('documents', lazy=True))
    
    def __repr__(self):
        return f"<Document {self.original_filename}>"


class Paragraph(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey('document.id'), nullable=False)
    text = db.Column(db.Text, nullable=False)
    index = db.Column(db.Integer, nullable=False)  # Order within document
    paragraph_type = db.Column(db.String(20), default="regular")  # regular, heading, list-item, table-cell
    
    # For similarity comparison
    similarity_hash = db.Column(db.String(128), nullable=True)
    
    # Relationships
    tags = db.relationship('Tag', secondary=paragraph_tags, lazy='subquery',
                          backref=db.backref('paragraphs', lazy=True))
    
    def __repr__(self):
        return f"<Paragraph {self.id} - Document {self.document_id}>"


class Tag(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False, unique=True)
    color = db.Column(db.String(7), nullable=False, default="#17a2b8")  # Default color
    
    def __repr__(self):
        return f"<Tag {self.name}>"
