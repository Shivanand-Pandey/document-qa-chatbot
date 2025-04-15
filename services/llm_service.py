import logging
import requests
import json
from typing import Dict, Any, List
import config

logger = logging.getLogger(__name__)

class LLMService:
    """Service for interacting with LLM models for summarization and RAG-based Q&A."""
    
    def __init__(self):
        self.model = config.LLM_MODEL
        self.endpoint = config.LLM_ENDPOINT
        logger.info(f"Initialized LLM service with model: {self.model}")
        
    async def generate_summary(self, text: str) -> str:
        """
        Generate a summary of the provided text.
        
        Args:
            text: The text to summarize
            
        Returns:
            str: The generated summary
        """
        if not text or not text.strip():
            return "No text provided for summarization."
            
        # Create the prompt for summarization
        prompt = config.SUMMARIZATION_PROMPT_TEMPLATE.format(text=text)
        
        try:
            # Call the LLM API
            response = await self._generate_text(prompt)
            return response
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Failed to generate summary due to an error."
    
    async def answer_question(self, question: str, context: List[Dict[str, Any]]) -> str:
        """
        Answer a question based on the provided context.
        
        Args:
            question: The user's question
            context: List of relevant text chunks
            
        Returns:
            str: The generated answer
        """
        if not question or not question.strip():
            return "No question provided."
            
        if not context:
            return "I don't have enough information to answer this question based on the document."
            
        # Concatenate context chunks
        context_text = "\n\n".join([f"Chunk {i+1}:\n{chunk['text']}" for i, chunk in enumerate(context)])
        
        # Create the prompt for Q&A
        prompt = config.RAG_PROMPT_TEMPLATE.format(
            context=context_text,
            question=question
        )
        
        try:
            # Call the LLM API
            response = await self._generate_text(prompt)
            return response
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return "Failed to generate an answer due to an error."
    
    async def _generate_text(self, prompt: str) -> str:
        """
        Generate text using the LLM API.
        
        Args:
            prompt: The prompt for the LLM
            
        Returns:
            str: The generated text
        """
        # For Ollama API
        try:
            payload = {
                "model": self.model.split('/')[-1] if '/' in self.model else self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 1024
                }
            }
            
            logger.debug(f"Sending request to LLM API: {self.endpoint}/api/generate")
            
            response = requests.post(
                f"{self.endpoint}/api/generate",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "No response from LLM")
            else:
                logger.error(f"LLM API error: {response.status_code} - {response.text}")
                return f"Error: LLM API returned status code {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            raise
