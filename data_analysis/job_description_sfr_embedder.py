import os
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from embeddings.sfr_embed_endpoint import SFREmbeddingEndpoint
from ai_systems.utils.exceptions import ModelServiceError
import random
import logging
import time
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('job_description_embedding.log')
    ]
)
logger = logging.getLogger(__name__)

def format_time(seconds):
    """Convert seconds to human readable time format"""
    return str(timedelta(seconds=int(seconds)))

def create_embeddings(input_dir: str, output_dir: str):
    start_time = time.time()
    logger.info(f"Starting embedding generation process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    random.shuffle(txt_files)
    total_files = len(txt_files)
    
    logger.info(f"Found {total_files} files to process")
    
    embedding_model = SFREmbeddingEndpoint()
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    pbar = tqdm(txt_files, desc="Processing Files", total=total_files)
    
    for filename in pbar:
        try:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace('.txt', '.npy'))
            
            if os.path.exists(output_path):
                skipped_count += 1
                continue
            
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            if not text:
                logger.warning(f"Skipping empty file: {filename}")
                error_count += 1
                continue
            
            embedding = embedding_model.get_embedding(text)
            np.save(output_path, embedding)
            processed_count += 1
            
            # Update progress bar statistics
            elapsed_time = time.time() - start_time
            files_per_second = processed_count / elapsed_time if elapsed_time > 0 else 0
            remaining_files = total_files - (processed_count + skipped_count + error_count)
            eta = remaining_files / files_per_second if files_per_second > 0 else 0
            
            pbar.set_postfix({
                'Processed': processed_count,
                'Skipped': skipped_count,
                'Errors': error_count,
                'ETA': format_time(eta)
            })
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            error_count += 1
            continue
    
    total_time = time.time() - start_time
    completion_rate = (processed_count + skipped_count) / total_files * 100
    
    logger.info("\nProcessing completed!")
    logger.info(f"Final statistics:")
    logger.info(f"- Total time: {format_time(total_time)}")
    logger.info(f"- Processed: {processed_count} new files")
    logger.info(f"- Skipped: {skipped_count} existing files")
    logger.info(f"- Errors: {error_count} files")
    logger.info(f"- Processing rate: {processed_count / total_time:.2f} files/second")
    logger.info(f"- Total completion rate: {completion_rate:.2f}%")
    
    print("\nEmbedding generation completed!")
    print(f"Total time: {format_time(total_time)}")
    print(f"Check job_description_embedding.log for detailed processing information")

if __name__ == "__main__":
    input_dir = "data/job_descriptions/format_txt"
    output_dir = "data/job_descriptions/sfr_embedding"
    create_embeddings(input_dir, output_dir)