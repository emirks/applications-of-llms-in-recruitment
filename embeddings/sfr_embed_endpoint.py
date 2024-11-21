from . import BaseEmbedding
from ai_systems.utils.utils import BaseValidator
from ai_systems.utils.exceptions import ModelServiceError

import requests
import logging
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HF_TOKEN = os.environ['DEV_HUGGINGFACE_TOKEN']
API_URL = "https://ifrjig6by5tj04vj.us-east-1.aws.endpoints.huggingface.cloud"
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}

class SFREmbeddingEndpoint(BaseEmbedding):
    def __init__(self, api_url=API_URL, headers=HEADERS):
        self.api_url = api_url
        self.headers = headers
        self.validator = BaseValidator()
        self.cache = {}
        logger.info("SFREmbeddingEndpoint initialized with API URL: %s", self.api_url)

    def get_embedding(self, text: str):
        """Get embedding using SFR-Embedding-2_R model"""
        try:
            self.validator.type_check(obj=text, obj_type=str, obj_name='text')
            
            cache_key = hash(text)
            if cache_key in self.cache:
                return self.cache[cache_key]

            payload = {
                "inputs": text
            }
            
            logger.info("Sending request to SFR API for text embedding")
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                embedding = response.json()
                if isinstance(embedding, list) and len(embedding) > 0:
                    embedding_array = np.array(embedding)
                    self.cache[cache_key] = embedding_array
                    return embedding_array
                else:
                    raise ModelServiceError(f"Unexpected response format: {embedding}")
            else:
                raise ModelServiceError(f"API error: {response.status_code}, {response.text}")

        except Exception as e:
            logger.error(f"Error during embedding: {str(e)}")
            raise ModelServiceError(f"Embedding generation failed: {str(e)}")