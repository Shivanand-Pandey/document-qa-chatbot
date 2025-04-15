import logging
import os
import sys
from pathlib import Path
import uvicorn
from fastapi import FastAPI
from ui.gradio_app import GradioInterface
import gradio as gr
import config
import webbrowser


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Intelligent Document Processing & Q&A System",
    description="A system for processing documents and answering questions",
    version="1.0.0"
)

# Initialize Gradio interface
gradio_interface = GradioInterface()
gradio_app = gradio_interface.create_ui()

# Mount Gradio app to FastAPI
app = gr.mount_gradio_app(app, gradio_app, path="/")

# Copy sample document if it doesn't exist
def setup_sample_document():
    sample_doc_path = config.ASSETS_DIR / "The_Gift_of_the_Magi.pdf"
    if not sample_doc_path.exists():
        logger.info("Sample document not found, please place 'The_Gift_of_the_Magi.pdf' in the assets directory")


if __name__ == "__main__":
    # Ensure necessary directories exist
    config.ASSETS_DIR.mkdir(exist_ok=True)
    config.TEMP_DIR.mkdir(exist_ok=True)
    config.VECTOR_DB_PATH.mkdir(exist_ok=True)
    
    # Setup sample document
    setup_sample_document()
    
    # Run the FastAPI application with Uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")

    # Open the app in the default web browser
    webbrowser.open(f"http://127.0.0.1:{port}")

    uvicorn.run(app, host="0.0.0.0", port=port)

