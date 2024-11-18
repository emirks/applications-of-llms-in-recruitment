import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from embeddings.b1ade_embed import B1adeEmbed

def create_embeddings(input_dir: str, output_dir: str, batch_size: int = 32):
    """
    Create embeddings for all job descriptions in the input directory
    
    Args:
        input_dir (str): Directory containing job description text files
        output_dir (str): Directory to save the embeddings
        batch_size (int): Number of files to process before freeing memory
    """
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get list of all txt files
    txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    total_files = len(txt_files)
    
    print(f"Found {total_files} files to process")
    
    # Initialize the embedding model
    embedding_model = B1adeEmbed()
    
    # Process files in batches
    for i in tqdm(range(0, total_files, batch_size)):
        batch_files = txt_files[i:i + batch_size]
        
        for filename in batch_files:
            try:
                # Get file paths
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename.replace('.txt', '.npy'))
                
                # Skip if embedding already exists
                if os.path.exists(output_path):
                    continue
                
                # Read the job description
                with open(input_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                
                # Skip empty files
                if not text:
                    continue
                
                # Get embedding
                embedding = embedding_model.get_embedding(text)
                
                # Save embedding
                np.save(output_path, embedding)
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
        

    # Final cleanup
    B1adeEmbed.free_memory()
    print("Embedding generation completed!")

if __name__ == "__main__":
    input_dir = "data/job_descriptions/format_txt"
    output_dir = "data/job_descriptions/format_embedding"
    
    create_embeddings(input_dir, output_dir, batch_size=32)