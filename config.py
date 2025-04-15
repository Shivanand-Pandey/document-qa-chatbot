import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
TEMP_DIR = BASE_DIR / "temp"

# Ensure directories exist
ASSETS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "llama3"
LLM_ENDPOINT = "http://localhost:11434"


# Vector DB settings
VECTOR_DB_PATH = BASE_DIR / "vectordb"
VECTOR_DB_PATH.mkdir(exist_ok=True)

# OCR settings
OCR_CHUNK_SIZE = 1000  # Maximum characters to send for OCR at once

# Text processing settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128

# Prompts
SUMMARIZATION_PROMPT_TEMPLATE = """
You are a helpful assistant that creates concise and accurate summaries of documents. 
Below is a text extracted from a document. Please summarize it effectively, focusing on:
1. The main plot or key information
2. Important characters or entities
3. Significant themes or findings

Text to summarize:
{text}

Provide a summary that captures the essence of the document in at most 5 paragraphs.
"""

RAG_PROMPT_TEMPLATE = """
You are a helpful assistant answering questions about a document. 
Use ONLY the following context to answer the user's question. If you can't find the 
answer in the context, say "I don't have enough information to answer this question 
based on the document." Don't use any other knowledge.

Context:
{context}

User Question: {question}

Your Answer:
"""
