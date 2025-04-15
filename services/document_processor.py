import PyPDF2
from pdf2image import convert_from_path
from PIL import Image
import io
import logging
from pathlib import Path
import config

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Service for processing PDF documents."""
    
    def __init__(self):
        self.temp_dir = config.TEMP_DIR
        
    async def process_pdf(self, file_path):
        """
        Process a PDF file and convert it to images for OCR processing.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            dict: Dictionary containing the PDF metadata and images
        """
        logger.info(f"Processing PDF: {file_path}")
        
        # Ensure file exists
        if not Path(file_path).exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        try:
            # Extract text with PyPDF2 first
            extracted_text = self._extract_text_with_pypdf(file_path)
            
            # If text extraction is insufficient, convert to images
            images = self._convert_pdf_to_images(file_path)
            
            return {
                "file_path": file_path,
                "extracted_text": extracted_text,
                "images": images
            }
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise
            
    def _extract_text_with_pypdf(self, file_path):
        """Extract text from PDF using PyPDF2."""
        text_by_page = {}
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for i, page in enumerate(pdf_reader.pages):
                    text = page.extract_text() or ""
                    text_by_page[i] = text
                    
            return text_by_page
        except Exception as e:
            logger.error(f"Error extracting text with PyPDF2: {e}")
            return {}
            
    def _convert_pdf_to_images(self, file_path):
        """Convert PDF pages to images for OCR processing."""
        images_by_page = {}
        
        try:
            # Convert PDF to images
            pdf_images = convert_from_path(
                file_path,
                dpi=300,  # Higher DPI for better OCR
                output_folder=self.temp_dir,
                fmt="jpeg",
                thread_count=2
            )
            
            # Save and store images
            for i, image in enumerate(pdf_images):
                image_path = self.temp_dir / f"page_{i}.jpg"
                image.save(image_path, "JPEG")
                images_by_page[i] = str(image_path)
                
            return images_by_page
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            return {}
            
    def get_image_bytes(self, image_path):
        """Convert image to bytes for OCR API."""
        with Image.open(image_path) as img:
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            return img_byte_arr.getvalue()
