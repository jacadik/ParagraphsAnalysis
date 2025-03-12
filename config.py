# config.py
import os

# Flask configuration
SECRET_KEY = 'change-this-to-a-random-secret-key'
DEBUG = True

# Database configuration
SQLALCHEMY_DATABASE_URI = 'sqlite:///document_analyzer.db'
SQLALCHEMY_TRACK_MODIFICATIONS = False

# Upload configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
