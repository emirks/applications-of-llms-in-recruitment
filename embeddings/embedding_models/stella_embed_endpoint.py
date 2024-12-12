from .. import BaseEmbedding
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

load_dotenv(override=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HF_TOKEN = os.environ['DEV_HUGGINGFACE_TOKEN']
API_URL = "https://bddwua1qb0minh42.eu-west-1.aws.endpoints.huggingface.cloud"
HEADERS = {
    "Accept": "application/json",
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}

class StellaEndpoint(BaseEmbedding):
    def __init__(self, api_url=API_URL, headers=HEADERS, cache_dir=None):
        self.api_url = api_url
        self.headers = headers
        self.validator = BaseValidator()
        self.cache = {}
        logger.info("StellaEndpoint initialized with API URL: %s", self.api_url)

    def get_embedding(self, text: str, should_chunk: bool = False):
        """Get embedding with retry logic and error handling"""
        try:
            self.validator.type_check(obj=text, obj_type=str, obj_name='text')
            
            cache_key = hash(text)
            if cache_key in self.cache:
                return self.cache[cache_key]

            # Simplify the payload structure
            payload = {
                "inputs": text,  # Just send the text directly
                "parameters": {
                    "wait_for_model": True
                }
            }
            
            logger.info(f"Sending request with payload: {payload}")
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            logger.info(f"Response status: {response.status_code}")
            logger.info(f"Response content: {response.text}")
            
            if response.status_code == 200:
                try:
                    embedding = response.json()
                    if isinstance(embedding, list) and len(embedding) > 0:
                        # Convert to numpy array
                        embedding_array = np.array(embedding)
                        if np.any(np.isnan(embedding_array)):
                            raise ModelServiceError("API returned NaN values in embedding")
                        self.cache[cache_key] = embedding_array
                        return embedding_array
                    else:
                        raise ModelServiceError(f"Unexpected response format: {embedding}")
                except Exception as e:
                    logger.error(f"Error processing response: {str(e)}")
                    raise ModelServiceError(f"Failed to process API response: {str(e)}")
            elif response.status_code == 413:
                raise PayloadTooLargeError("Input text too large")
            else:
                raise ModelServiceError(f"API error: {response.status_code}, {response.text}")

        except Exception as e:
            logger.error(f"Error during embedding: {str(e)}")
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