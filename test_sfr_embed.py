from embeddings.sfr_embed_endpoint import SFREmbeddingEndpoint
import logging
import numpy as np

# Set logging to DEBUG level
logging.basicConfig(level=logging.DEBUG)

def test_embedding():
    model = SFREmbeddingEndpoint()
    
    # Test cases
    test_cases = [
        "This is a simple test sentence.",
        """This is a longer test paragraph that contains multiple sentences.
        It should test the model's ability to handle longer texts and maintain
        semantic meaning across sentences.""",
    ]
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\nTest case {i}:")
        print(f"Input text: {test_text[:100]}...")
        
        try:
            embedding = model.get_embedding(test_text)
            print(f"Embedding shape: {embedding.shape}")
            print(f"First few values: {embedding.flatten()[:5]}")
            print(f"Mean value: {np.mean(embedding):.6f}")
            print(f"Std deviation: {np.std(embedding):.6f}")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_embedding()