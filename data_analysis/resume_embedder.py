import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from embeddings.b1ade_embed import B1adeEmbed
from resume_parser.resume_parser_local import ResumeParser
import random

def create_embeddings(input_base_dir: str, output_base_dir: str, txt_base_dir: str, batch_size: int = 32):
    """
    Create embeddings for all resumes in the input directory, organized by category
    
    Args:
        input_base_dir (str): Base directory containing category folders with resume PDFs
        output_base_dir (str): Base directory to save the embeddings by category
        txt_base_dir (str): Base directory to save the extracted text by category
        batch_size (int): Number of files to process before freeing memory
    """
    
    # Initialize the embedding model and resume parser
    embedding_model = B1adeEmbed()
    resume_parser = ResumeParser()
    
    # Get all categories (subdirectories)
    categories = [d for d in os.listdir(input_base_dir) if os.path.isdir(os.path.join(input_base_dir, d))]
    
    total_processed = 0
    total_skipped = 0
    
    for category in categories:
        # Create output directories for embeddings and text
        category_output_dir = os.path.join(output_base_dir, category)
        category_txt_dir = os.path.join(txt_base_dir, category)
        Path(category_output_dir).mkdir(parents=True, exist_ok=True)
        Path(category_txt_dir).mkdir(parents=True, exist_ok=True)
        
        # Get list of all pdf files in this category
        input_dir = os.path.join(input_base_dir, category)
        pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]
        
        # Randomize the file order
        random.shuffle(pdf_files)
        total_files = len(pdf_files)
        
        print(f"\nProcessing category: {category}")
        print(f"Found {total_files} files to process")
        
        # Track processed files for this category
        processed_count = 0
        skipped_count = 0
        
        # Process files in batches
        for i in tqdm(range(0, total_files, batch_size)):
            batch_files = pdf_files[i:i + batch_size]
            
            for filename in batch_files:
                try:
                    # Get file paths
                    input_path = os.path.join(input_dir, filename)
                    output_path = os.path.join(category_output_dir, filename.replace('.pdf', '.npy'))
                    txt_path = os.path.join(category_txt_dir, filename.replace('.pdf', '.txt'))
                    
                    # Skip if both embedding and text already exist
                    if os.path.exists(output_path) and os.path.exists(txt_path):
                        skipped_count += 1
                        continue
                    
                    # Extract text using resume parser
                    text = resume_parser.extract_text(input_path)
                    
                    # Skip empty files
                    if not text:
                        continue
                    
                    # Save text file
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(text)
                    
                    # Get and save embedding
                    embedding = embedding_model.get_embedding(text)
                    np.save(output_path, embedding)
                    
                    processed_count += 1
                    
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    continue
                
        print(f"Category {category} completed! Processed {processed_count} new files, skipped {skipped_count} existing files.")
        total_processed += processed_count
        total_skipped += skipped_count
    
    print(f"\nAll categories completed!")
    print(f"Total processed: {total_processed} files")
    print(f"Total skipped: {total_skipped} files")
    
    # Final cleanup
    B1adeEmbed.free_memory()
    print("Embedding generation completed!")

if __name__ == "__main__":
    input_dir = "data/resume/format_pdf"
    output_dir = "data/resume/format_embedding"
    txt_dir = "data/resume/format_txt"
    
    create_embeddings(input_dir, output_dir, txt_dir, batch_size=32)