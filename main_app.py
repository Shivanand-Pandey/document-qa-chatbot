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



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


app = FastAPI(
    title="Intelligent Document Processing & Q&A System",
    description="A system for processing documents and answering questions",
    version="1.0.0"
)


gradio_interface = GradioInterface()
gradio_app = gradio_interface.create_ui()


app = gr.mount_gradio_app(app, gradio_app, path="/")



if __name__ == "__main__":

    config.ASSETS_DIR.mkdir(exist_ok=True)
    config.TEMP_DIR.mkdir(exist_ok=True)
    config.VECTOR_DB_PATH.mkdir(exist_ok=True)
    

    setup_sample_document()
    

    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")


    webbrowser.open(f"http://127.0.0.1:{port}")

    uvicorn.run(app, host="0.0.0.0", port=port)

