from . import BaseEmbedding
from ai_systems.utils.utils import BaseValidator
from ai_systems.utils.exceptions import ModelServiceError, PayloadTooLargeError, EmbeddingError

import requests
import logging
import os
import numpy as np
from dotenv import load_dotenv
from requests.exceptions import RequestException
from ai_systems.utils.exceptions import (
    EmbeddingError,
    ModelServiceError,
    PayloadTooLargeError,
    TokenExceededError
)
from typing import List
from datetime import datetime, timedelta

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HF_TOKEN = os.environ['DEV_HUGGINGFACE_TOKEN']
API_URL = "https://lw1bfncqv3sue1w0.us-east-1.aws.endpoints.huggingface.cloud"
HEADERS = {
    "Accept": "application/json",
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}

class MixedBreadLargeV1HuggingFaceEndpoint(BaseEmbedding):
    def __init__(self, api_url=API_URL, headers=HEADERS, cache_dir=None):
        self.api_url = api_url
        self.headers = headers
        self.validator = BaseValidator()
        self.cache = {}  # Simple in-memory cache
        logger.info("HuggingfaceEmbedder initialized with API URL: %s", self.api_url)

    def get_embedding(self, text: str, should_chunk: bool = False):
        """Get embedding with retry logic and error handling"""
        try:
            # Input validation
            self.validator.type_check(obj=text, obj_type=str, obj_name='text')
            
            # Check cache first
            cache_key = hash(text)
            if cache_key in self.cache:
                logger.debug("Cache hit for text")
                return self.cache[cache_key]

            # If previous attempt failed with PayloadTooLarge or text is long, chunk it
            if should_chunk:  # Conservative token estimate
                logger.info("Processing text in chunks due to length or previous error")
                chunks = self._chunk_text(text)
                chunk_embeddings = []
                
                for chunk in chunks:
                    payload = {"inputs": chunk, "parameters": {}}
                    response = requests.post(self.api_url, headers=self.headers, json=payload)
                    
                    if response.status_code == 200:
                        chunk_embeddings.append(np.array(response.json()))
                    elif response.status_code == 413:
                        logger.error(f"Chunk still too large: {len(chunk)} chars")
                        raise PayloadTooLargeError("Chunk still exceeds size limit")
                    else:
                        raise ModelServiceError(f"Chunk embedding failed with status code: {response.status_code}")
                
                if not chunk_embeddings:
                    raise EmbeddingError("No valid embeddings generated from chunks")
                    
                embeddings = np.mean(chunk_embeddings, axis=0)
                self.cache[cache_key] = embeddings
                return embeddings

            # Regular request without chunking
            logger.info("Sending request to Huggingface API for text embedding")
            payload = {"inputs": text, "parameters": {}}
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                embeddings = np.array(response.json())
                self.cache[cache_key] = embeddings
                return embeddings
            elif response.status_code == 413:
                logger.info("Payload too large, will retry with chunking")
                return self.get_embedding(text, should_chunk=True)
            elif response.status_code == 429:
                raise ModelServiceError("API rate limit exceeded")
            elif response.status_code == 503:
                raise ModelServiceError("Service temporarily unavailable")
            else:
                raise ModelServiceError(f"API error: {response.status_code}")

        except requests.exceptions.RequestException as e:
            raise ModelServiceError(f"Request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during embedding: {str(e)}")
            raise ModelServiceError(f"Embedding generation failed: {str(e)}")

    def _chunk_text(self, text: str, max_tokens: int = 512) -> List[str]:
        """Split text into chunks based on token limit
        
        Args:
            text: Text to split
            max_tokens: Maximum number of tokens per chunk (default: 512 for model limit)
        """
        # Rough approximation: 1 token ≈ 4 chars for English text
        approx_chars_per_chunk = max_tokens * 3
        
        # First split into sentences to avoid breaking mid-sentence
        sentences = text.replace('\n', ' ').split('.')
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip() + '.'  # Add period back
            sentence_length = len(sentence)
            
            if current_length + sentence_length <= approx_chars_per_chunk:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                # If current chunk has content, add it to chunks
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                # Start new chunk with current sentence
                current_chunk = [sentence]
                current_length = sentence_length
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks

# Example usage
if __name__ == "__main__":
    embedder = MixedBreadLargeV1HuggingFaceEndpoint()
    output = embedder.get_embedding("This sound track was beautiful! It paints the scenery in your mind so well I would recommend it even to people who hate video game music!")
    print(output)
