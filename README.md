# 📄 AI Document Q&A Chatbot

This project is a complete implementation of a document processing pipeline that extracts content from scanned PDFs, summarizes them, and answers user queries via a chatbot using retrieval-augmented generation (RAG).

---

## 🎯 Assignment Objective
Design and build an **LLM-based system** to:
- Upload and preprocess scanned documents
- Extract key information using **OCR**
- Summarize content and enable **Q&A** via **semantic search** and **LLMs**
- Package the tool with a simple **web UI** and a clear README

---

## ✅ Implemented Features

### 🔍 Document Understanding
- Accepts **PDF uploads**
- Performs **OCR** with Tesseract for scanned pages
- Extracts **title** and **first line** for metadata

### 🧠 Embedding & Chunking
- Chunks the document into paragraphs
- Embeds using `BAAI/bge-small-en-v1.5`
- Stores vector embeddings in **ChromaDB**

### 🤖 Q&A with RAG
- Uses local **LLM (LLaMA3)** via **Ollama**
- Retrieves relevant chunks with semantic search
- Formats prompt and generates answers with context

### 🖥️ Web Interface
- Built using **Gradio**
- Upload documents and chat on the **same page**
- No external APIs required — runs **fully local**

---

## 🧪 Sample Interaction

> **User**: What’s the title of the story?  
> **Bot**: The title of the story is: **The Gift of the Magi**

> **User**: Who are the main characters?  
> **Bot**: Della and Jim are the main characters, introduced early in the story...

> **User**: Provide summary of the story.  
> **Bot**: Based on the provided context (Chunks 1–5), here's a summary...

---

## 🧰 Tech Stack

| Component      | Tool/Lib                          |
|----------------|------------------------------------|
| OCR            | Tesseract                         |
| Chunking/Embed | `sentence-transformers` + BGE     |
| Vector DB      | Chroma                            |
| LLM            | Ollama + Meta `llama3`            |
| UI             | Gradio                            |

---

## 🛠️ Setup Instructions

### 📦 Dependencies
```bash
pip install -r requirements.txt
```

### 📥 Install Tesseract OCR
- Windows: https://github.com/tesseract-ocr/tesseract
- Add Tesseract to system PATH

### 📥 Install Poppler (PDF to image)
- https://github.com/oschwartz10612/poppler-windows/releases
- Add `poppler/bin` to your system PATH

### 🧠 Install Ollama + LLaMA3
```bash
ollama run llama3
```

---

## 🚀 How to Run
```bash
python main_app.py
```
Then open: [http://localhost:8000](http://localhost:8000)

---

## 📁 Folder Structure
```
chatbot-app/
├── main_app.py              # Entrypoint (FastAPI + Gradio)
├── config.py                # Paths and model settings
├── requirements.txt         # Python dependencies
├── assets/                  # Sample PDF(s)
├── services/                # Core logic for OCR, LLM, Embedding
├── ui/                      # Gradio UI logic
├── vectordb/                # ChromaDB storage
├── temp/                    # OCR image cache
```

---
