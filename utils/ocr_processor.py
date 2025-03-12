# utils/ocr_processor.py
import os
import re
import fitz  # PyMuPDF
import tempfile
import numpy as np
from PIL import Image, ImageEnhance
from pdf2image import convert_from_path
import pytesseract

class OCRProcessor:
    """Class for processing scanned documents with OCR."""
    
    def __init__(self, language='eng'):
        """
        Initialize OCR processor.
        
        Args:
            language: OCR language code(s) to use (e.g., 'eng' or 'eng+fra')
        """
        self.language = language
        self.min_confidence = 60  # Minimum confidence for OCR to accept
    
    def is_scanned_pdf(self, pdf_path, sample_pages=3):
        """
        Determine if a PDF is likely scanned or contains embedded text.
        
        Args:
            pdf_path: Path to the PDF file
            sample_pages: Number of pages to sample for checking
            
        Returns:
            bool: True if PDF appears to be scanned, False if it has embedded text
        """
        doc = fitz.open(pdf_path)
        
        # Sample a few pages to check for text
        pages_to_check = min(sample_pages, len(doc))
        total_text = 0
        
        for page_num in range(pages_to_check):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            total_text += len(text.strip())
        
        # If very little text is detected, assume it's a scanned document
        return total_text < 100 * pages_to_check
    
    def preprocess_image(self, image):
        """
        Preprocess image for better OCR results.
        
        Args:
            image: PIL Image object
            
        Returns:
            PIL Image: Preprocessed image
        """
        # Convert to grayscale if color
        if image.mode != 'L':
            image = image.convert('L')
        
        # Increase contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.5)
        
        # Denoise (alternative: use opencv with cv2.fastNlMeansDenoising)
        # This is a simple approach; more advanced denoising could be applied
        
        return image
    
    def perform_ocr(self, image):
        """
        Perform OCR on a preprocessed image.
        
        Args:
            image: PIL Image object
            
        Returns:
            str: Extracted text
            dict: Additional OCR data
        """
        # Get text using pytesseract
        custom_config = f'--oem 3 --psm 6 -l {self.language}'
        text = pytesseract.image_to_string(image, config=custom_config)
        
        # Get detailed OCR data including confidence scores
        data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
        
        return text, data
    
    def extract_text_with_layout(self, pdf_path, output_format='paragraphs'):
        """
        Extract text from scanned PDF with layout awareness.
        
        Args:
            pdf_path: Path to the PDF file
            output_format: Format to return ('paragraphs' or 'raw')
            
        Returns:
            list: Text extracted from the document
        """
        # Check if PDF is actually scanned
        if not self.is_scanned_pdf(pdf_path):
            raise ValueError("This PDF appears to contain embedded text. Use PyMuPDF instead of OCR.")
        
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=300)
        
        extracted_data = []
        
        # Process each page
        for i, img in enumerate(images):
            # Preprocess image
            preprocessed = self.preprocess_image(img)
            
            # Perform OCR with detailed data
            _, ocr_data = self.perform_ocr(preprocessed)
            
            # Skip low confidence text
            filtered_text = []
            current_block_text = []
            current_block_num = -1
            
            for j in range(len(ocr_data['text'])):
                text = ocr_data['text'][j]
                conf = int(ocr_data['conf'][j])
                block_num = ocr_data['block_num'][j]
                
                # Skip if confidence is too low or text is empty
                if conf < self.min_confidence or not text.strip():
                    continue
                
                # If we're in a new block, add the previous block to results
                if block_num != current_block_num and current_block_text:
                    block_content = ' '.join(current_block_text)
                    filtered_text.append(block_content)
                    current_block_text = []
                
                current_block_text.append(text)
                current_block_num = block_num
            
            # Add the last block
            if current_block_text:
                block_content = ' '.join(current_block_text)
                filtered_text.append(block_content)
            
            # Organize text into paragraphs if requested
            if output_format == 'paragraphs':
                paragraphs = self._organize_into_paragraphs(filtered_text)
                extracted_data.append({
                    'page': i + 1,
                    'paragraphs': paragraphs
                })
            else:
                extracted_data.append({
                    'page': i + 1,
                    'text': '\n'.join(filtered_text)
                })
        
        return extracted_data
    
    def _organize_into_paragraphs(self, text_blocks):
        """
        Organize text blocks into paragraphs.
        
        Args:
            text_blocks: List of text blocks from OCR
            
        Returns:
            list: List of paragraphs with their metadata
        """
        paragraphs = []
        current_paragraph = ""
        
        for block in text_blocks:
            block = block.strip()
            if not block:
                continue
            
            # Determine block type
            is_heading = False
            is_list_item = False
            
            # Check for headings (all caps, short text, or ending with colon)
            if (block.isupper() and len(block) < 100) or \
               (len(block) < 50 and block.endswith(':')) or \
               re.match(r'^[A-Z0-9][A-Z0-9\.\s]{0,30}$', block):
                is_heading = True
            
            # Check for list items
            if re.match(r'^\s*[\•\-\*\d]+[\.\)\]]\s', block):
                is_list_item = True
            
            # Determine paragraph type
            if is_heading:
                para_type = "heading"
            elif is_list_item:
                para_type = "list-item"
            else:
                para_type = "regular"
            
            # If current paragraph is not empty and this block is a different type,
            # save the current paragraph and start a new one
            if current_paragraph and (is_heading or is_list_item):
                paragraphs.append({
                    'text': current_paragraph,
                    'type': 'regular'
                })
                current_paragraph = ""
            
            # If it's a special type, add it as its own paragraph
            if is_heading or is_list_item:
                paragraphs.append({
                    'text': block,
                    'type': para_type
                })
            else:
                # Check if we should start a new paragraph or continue the current one
                if current_paragraph:
                    # Check if current paragraph ends with sentence-ending punctuation
                    if current_paragraph[-1] in '.!?':
                        current_paragraph += " " + block
                    else:
                        # If not, this might be a continuation of a line broken by OCR
                        current_paragraph += " " + block
                else:
                    current_paragraph = block
        
        # Add the last paragraph if any
        if current_paragraph:
            paragraphs.append({
                'text': current_paragraph,
                'type': 'regular'
            })
        
        return paragraphs
    
    def process_document(self, file_path):
        """
        Process a document with OCR and return structured paragraphs.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            list: List of (text, paragraph_type) tuples
        """
        # Extract text with layout
        extracted_data = self.extract_text_with_layout(file_path)
        
        # Combine pages into unified paragraph list
        result = []
        for page_data in extracted_data:
            for para in page_data['paragraphs']:
                result.append((para['text'], para['type']))
        
        return result
