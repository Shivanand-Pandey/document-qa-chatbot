import logging
import pytesseract
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)

class OCRService:
    """Service for performing OCR on document images using Tesseract."""
    
    def __init__(self):
        # Optional: point to the Tesseract executable if it's not in PATH
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        pass
        
    async def process_images(self, images_by_page):
        """
        Process images and extract text using Tesseract OCR.

        Args:
            images_by_page: Dictionary of page numbers and image paths

        Returns:
            dict: Dictionary of page numbers and extracted text
        """
        logger.info(f"[OCR] Processing {len(images_by_page)} pages with Tesseract")

        extracted_text = {}

        for page_num, image_path in images_by_page.items():
            try:
                text = self._extract_text_from_image(image_path)
                extracted_text[page_num] = text
                logger.info(f"[OCR] Page {page_num} text length: {len(text)}")
            except Exception as e:
                logger.error(f"[OCR] Error on page {page_num}: {e}")
                extracted_text[page_num] = ""

        return extracted_text

    def _extract_text_from_image(self, image_path):
        """Extract text from a single image using Tesseract."""
        image_path = Path(image_path)
        
        if not image_path.exists():
            logger.error(f"[OCR] Image file not found: {image_path}")
            return ""

        try:
            with Image.open(image_path) as img:
                text = pytesseract.image_to_string(img)
                return text
        except Exception as e:
            logger.error(f"[OCR] Failed to OCR image: {e}")
            return ""
