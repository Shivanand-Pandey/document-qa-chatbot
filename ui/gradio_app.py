import gradio as gr
import logging
from pathlib import Path
import os
import asyncio
import time

# ✅ Import all required services
import config
from services.document_processor import DocumentProcessor
from services.ocr_service import OCRService
from services.embedding_service import EmbeddingService
from services.vector_db_service import VectorDBService
from services.llm_service import LLMService


class GradioInterface:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.ocr_service = OCRService()
        self.embedding_service = EmbeddingService()
        self.vector_db_service = VectorDBService(self.embedding_service)
        self.llm_service = LLMService()

        # Tracking state
        self.current_document = None
        self.document_summary = None
        self.collection_name = None
        self.document_metadata = {}

    def create_ui(self):
        """Create and configure the Gradio UI."""
        with gr.Blocks(title="Intelligent Document Processing & Q&A") as app:
            gr.Markdown("# 📄 Intelligent Document Processing & Q&A System")

            with gr.Row():
                with gr.Column(scale=2):
                    file_input = gr.File(
                        label="Upload PDF Document",
                        file_types=[".pdf"],
                        type="file"
                    )
                    process_btn = gr.Button("Process Document", variant="primary")
                    status_output = gr.Textbox(label="Processing Status", interactive=False)

                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(height=500)

            with gr.Row():
                question_input = gr.Textbox(
                    label="Ask a question about the document",
                    placeholder="Type your question here...",
                    lines=2
                )
                submit_btn = gr.Button("Submit", variant="primary")

            with gr.Row():
                clear_btn = gr.Button("Clear Chat")

            process_btn.click(
                fn=self.process_document,
                inputs=[file_input],
                outputs=[status_output]
            )

            submit_btn.click(
                fn=self.answer_question,
                inputs=[question_input, chatbot],
                outputs=[chatbot, question_input]
            )

            question_input.submit(
                fn=self.answer_question,
                inputs=[question_input, chatbot],
                outputs=[chatbot, question_input]
            )

            clear_btn.click(
                fn=lambda: None,
                inputs=None,
                outputs=chatbot,
                show_progress=False
            )

        return app

    async def _process_pdf_async(self, file_path):
        """Process the PDF file asynchronously."""
        try:
            processed_doc = await self.document_processor.process_pdf(file_path)
            extracted_text = ""

            for page_num in sorted(processed_doc["extracted_text"].keys()):
                page_text = processed_doc["extracted_text"][page_num]
                if page_text and page_text.strip():
                    extracted_text += page_text + "\n\n"

            if len(extracted_text.strip()) < 100 and processed_doc["images"]:
                ocr_text = await self.ocr_service.process_images(processed_doc["images"])
                for page_num in sorted(ocr_text.keys()):
                    extracted_text += ocr_text[page_num] + "\n\n"

            # Extract title and first line
            lines = extracted_text.strip().splitlines()
            non_empty_lines = [line.strip() for line in lines if line.strip()]
            title_line = non_empty_lines[0] if non_empty_lines else "Unknown"
            first_line = non_empty_lines[1] if len(non_empty_lines) > 1 else "Unknown"
            self.document_metadata = {
                "title": title_line,
                "first_line": first_line
            }

            filename = Path(file_path).stem
            self.collection_name = f"doc_{filename}_{int(time.time())}"

            chunks = self.embedding_service.chunk_text(extracted_text)
            self.vector_db_service.create_collection(self.collection_name, overwrite=True)
            self.vector_db_service.add_documents(self.collection_name, chunks)

            self.current_document = {
                "path": file_path,
                "filename": filename,
                "text": extracted_text,
                "chunks": chunks
            }

            return True, "Document processed successfully! Ready for Q&A."
        except Exception as e:
            logging.error(f"Error processing document: {e}")
            return False, f"Error processing document: {str(e)}"

    def process_document(self, file_obj):
        """Process the uploaded document."""
        if not file_obj:
            return "Please upload a PDF file first."

        try:
            file_path = file_obj.name
            success, result = asyncio.run(self._process_pdf_async(file_path))
            return result
        except Exception as e:
            logging.error(f"Error in document processing: {e}")
            return f"Error: {str(e)}"

    async def _answer_question_async(self, question):
        """Answer a question asynchronously using RAG."""
        if not self.collection_name:
            return "Please upload and process a document first."

        results = self.vector_db_service.query_collection(
            self.collection_name,
            question,
            n_results=5
        )

        answer = await self.llm_service.answer_question(question, results)
        return answer

    def answer_question(self, question, history):
        """Handle question answering in the chatbot."""
        if not question or not question.strip():
            return history, ""

        if not self.collection_name:
            history.append((question, "Please upload and process a document first."))
            return history, ""

        lower_q = question.lower().strip()

        # Only respond with title directly if the question is very specific
        if any(kw in lower_q for kw in ["title", "name of the story", "chapter title"]) and self.document_metadata.get("title"):
            answer = f"The title of the story is: **{self.document_metadata['title']}**"
            history.append((question, answer))
            return history, ""

        if "first line" in lower_q and self.document_metadata.get("first_line"):
            answer = f"The first line of the story is: \"{self.document_metadata['first_line']}\""
            history.append((question, answer))
            return history, ""

        try:
            answer = asyncio.run(self._answer_question_async(question))
            history.append((question, answer))
            return history, ""
        except Exception as e:
            logging.error(f"Error answering question: {e}")
            history.append((question, f"Error: {str(e)}"))
            return history, ""