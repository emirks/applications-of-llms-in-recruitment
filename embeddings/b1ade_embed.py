from . import BaseEmbedding
from ai_systems.utils.utils import BaseValidator
import torch
from transformers import AutoTokenizer, AutoModel
import gc  # For garbage collection on CPU

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_NAME = 'w601sxs/b1ade-embed'
EMBEDDING_DIM = 1024

class B1adeEmbed(BaseEmbedding):
    # Class-level variables to store the model and tokenizer
    model = None
    tokenizer = None

    def __init__(self, model_name=MODEL_NAME):
        logger.info("Initializing B1adeEmbed with model: %s", model_name)

        # Check if the model is already initialized
        if B1adeEmbed.model is None or B1adeEmbed.tokenizer is None:
            logger.info("Model or tokenizer not found in memory, loading new instance.")
            B1adeEmbed.tokenizer = AutoTokenizer.from_pretrained(model_name)
            B1adeEmbed.model = AutoModel.from_pretrained(model_name)
            B1adeEmbed.model.eval()  # Set the model to evaluation mode
            
            # Move the model to GPU if available
            if torch.cuda.is_available():
                logger.info("Using device: cuda")
                B1adeEmbed.model.cuda()
            else:
                logger.info("CUDA not available, using CPU")
        else:
            logger.info("Reusing the previously loaded model and tokenizer from memory.")

        self.tokenizer = B1adeEmbed.tokenizer
        self.model = B1adeEmbed.model
        self.validator = BaseValidator()
        self.embedding_dim = EMBEDDING_DIM
        logger.info("Embedding Dimension: %d", self.embedding_dim)

    def get_embedding(self, text: str) -> torch.Tensor:
        """
        Get the embedding of a sentence using b1ade_embed.
        Default embedding dimension: 1024

        :param text: A string representing the sentence.
        :return: A tensor representing the sentence embedding.
        """
        logger.debug("Validating input text for get_embedding method")
        self.validator.type_check(obj=text, obj_type=str, obj_name='text')
        
        # Tokenize input text and get tensor
        logger.debug("Tokenizing input text")
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=512  # Adjust to model's actual max length if needed
        )

        # Move inputs to GPU if available
        if torch.cuda.is_available():
            logger.debug("Moving inputs to GPU")
            inputs = {key: tensor.cuda() for key, tensor in inputs.items()}

        # Get hidden states from the model
        logger.debug("Generating embeddings from the model")
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logger.debug("Returning the mean of the hidden states as the sentence embedding")
        # Return the mean of the hidden states as the sentence embedding
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    @classmethod
    def free_memory(cls):
        """
        Free up the model from GPU or CPU memory.
        """
        logger.info("Freeing up the embedding model from memory")

        # Check if the class attributes 'model' and 'tokenizer' are present
        if hasattr(cls, 'model') and cls.model is not None:
            # Free GPU memory if the model is on CUDA
            if cls.model.device.type == 'cuda':
                logger.info("Freeing model from GPU memory")
                del cls.model  # Remove the model from GPU memory
                torch.cuda.empty_cache()  # Clear the GPU cache
            else:
                # Free CPU memory
                logger.info("Freeing model from CPU memory")
                del cls.model  # Remove the model from CPU memory
                gc.collect()  # Force garbage collection for CPU

            # Reset the class-level variables
            cls.model = None
            cls.tokenizer = None
            logger.info("Model and tokenizer successfully freed")
        else:
            logger.warning("Model not initialized. Nothing to free.")