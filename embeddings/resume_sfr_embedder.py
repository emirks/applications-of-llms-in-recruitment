import os
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from embeddings.embedding_models.sfr_embed_endpoint import SFREmbeddingEndpoint
from ai_systems.utils.exceptions import ModelServiceError
import random
import logging
import time
from datetime import datetime, timedelta
import concurrent.futures
import threading
import signal
import sys
import backoff
from typing import List, Tuple
from dataclasses import dataclass
from requests.exceptions import RequestException

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('resume_sfr_embedding.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class EmbeddingTask:
    filename: str
    category: str
    input_path: str
    output_path: str
    text: str
    retries: int = 0
    max_retries: int = 3

# Global variables
stop_processing = False
counter_lock = threading.Lock()

def signal_handler(signum, frame):
    global stop_processing
    print("\nGraceful shutdown initiated. Waiting for current tasks to complete...")
    stop_processing = True

# Register both SIGINT (Ctrl+C) and SIGTERM
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def format_time(seconds):
    """Convert seconds to human readable time format"""
    return str(timedelta(seconds=int(seconds)))

@backoff.on_exception(
    backoff.expo,
    (RequestException, ModelServiceError),
    max_tries=3,
    giveup=lambda e: isinstance(e, ModelServiceError) and "rate limit" in str(e).lower()
)
def get_embedding_with_retry(model: SFREmbeddingEndpoint, text: str) -> np.ndarray:
    """Get embedding with exponential backoff retry"""
    try:
        return model.get_embedding(text)
    except ModelServiceError as e:
        if "must have less than 4096 tokens" in str(e):
            logger.error(f"Text too long ({len(text)} chars), should have been caught by split_text")
            raise
        raise

def split_text(text: str, max_tokens: int = 2500) -> List[str]:
    """Split text into chunks that fit within token limit
    Using very conservative estimates:
    - Average English word is ~5 characters
    - Average token is ~3 characters
    - Add extra buffer for safety
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    # Very conservative: 3 chars per token
    max_chars = max_tokens * 3  
    
    for word in words:
        word_length = len(word) + 1  # +1 for space
        if current_length + word_length > max_chars:
            # Save current chunk
            chunks.append(' '.join(current_chunk))
            # Start new chunk
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length
    
    # Add the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    # Validate chunk sizes
    for i, chunk in enumerate(chunks):
        chunk_len = len(chunk)
        if chunk_len > max_chars:
            logger.warning(f"Chunk {i} is still too large ({chunk_len} chars). Further splitting needed.")
            # Split this chunk again recursively
            subchunks = split_text(chunk, max_tokens=max_tokens//2)
            # Replace the large chunk with its subchunks
            chunks[i:i+1] = subchunks
    
    return chunks

def process_file(task: EmbeddingTask, embedding_model: SFREmbeddingEndpoint) -> Tuple[str, bool, str]:
    """Process a single file and return results"""
    try:
        # Split text into chunks
        chunks = split_text(task.text)
        
        if len(chunks) > 1:
            logger.info(f"Split {task.category}/{task.filename} into {len(chunks)} chunks")
            
            # Get embeddings for each chunk
            embeddings = []
            for i, chunk in enumerate(chunks):
                try:
                    embedding = get_embedding_with_retry(embedding_model, chunk)
                    embeddings.append(embedding)
                    
                    # Save individual chunk embedding
                    chunk_path = task.output_path.replace('.npy', f'_chunk{i}.npy')
                    np.save(chunk_path, embedding)
                except Exception as e:
                    logger.error(f"Error processing chunk {i} of {task.filename}: {str(e)}")
                    continue
            
            if not embeddings:
                return task.filename, False, "All chunks failed to process"
            
            # Calculate and save average embedding
            avg_embedding = np.mean(embeddings, axis=0)
            np.save(task.output_path, avg_embedding)
            
            return task.filename, True, ""
        else:
            # Single chunk case
            embedding = get_embedding_with_retry(embedding_model, chunks[0])
            np.save(task.output_path, embedding)
            return task.filename, True, ""
            
    except Exception as e:
        error_msg = f"Failed after {task.max_retries} retries: {str(e)}"
        return task.filename, False, error_msg

def create_embedding_tasks(input_base_dir: str, output_base_dir: str) -> Tuple[List[EmbeddingTask], int]:
    """Create list of embedding tasks from input files"""
    tasks = []
    skipped_count = 0
    categories = [d for d in os.listdir(input_base_dir) 
                 if os.path.isdir(os.path.join(input_base_dir, d))]
    
    for category in categories:
        category_path = os.path.join(input_base_dir, category)
        output_category_path = os.path.join(output_base_dir, category)
        Path(output_category_path).mkdir(parents=True, exist_ok=True)
        
        for filename in os.listdir(category_path):
            if not filename.endswith('.txt'):
                continue
                
            input_path = os.path.join(category_path, filename)
            output_path = os.path.join(output_category_path, filename.replace('.txt', '.npy'))
            
            if os.path.exists(output_path):
                skipped_count += 1
                continue
                
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                
                if text:
                    tasks.append(EmbeddingTask(
                        filename=filename,
                        category=category,
                        input_path=input_path,
                        output_path=output_path,
                        text=text
                    ))
            except Exception as e:
                logger.error(f"Error reading {filename} in {category}: {str(e)}")
                
    return tasks, skipped_count

def create_embeddings(input_dir: str, output_dir: str, max_workers: int = 100):
    """Create embeddings using multiple workers"""
    signal.signal(signal.SIGINT, signal_handler)
    
    start_time = time.time()
    logger.info(f"Starting resume embedding process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    tasks, skipped_count = create_embedding_tasks(input_dir, output_dir)
    total_files = len(tasks)
    random.shuffle(tasks)
    
    logger.info(f"Found {total_files} files to process")
    
    processed_count = 0
    error_count = 0
    retry_count = 0
    
    pbar = tqdm(total=total_files, desc="Processing Resumes")
    
    def process_task(task: EmbeddingTask) -> Tuple[bool, str]:
        if stop_processing:
            return False, "Interrupted by user"
            
        try:
            model = SFREmbeddingEndpoint()
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            return False, f"Model initialization failed: {str(e)}"
        
        filename, success, error = process_file(task, model)
        
        nonlocal processed_count, error_count, retry_count
        with counter_lock:
            if success:
                processed_count += 1
                logger.info(f"Successfully processed {task.category}/{task.filename}")
            else:
                if task.retries < task.max_retries:
                    task.retries += 1
                    retry_count += 1
                    logger.warning(f"Retrying {task.category}/{task.filename} (attempt {task.retries}/{task.max_retries})")
                    return False, "Requeued for retry"
                else:
                    error_count += 1
                    logger.error(f"Error processing {task.category}/{task.filename}: {error}")
            
            pbar.update(1)
            elapsed_time = time.time() - start_time
            files_per_second = processed_count / elapsed_time if elapsed_time > 0 else 0
            remaining_files = total_files - (processed_count + error_count)
            eta = remaining_files / files_per_second if files_per_second > 0 else 0
            
            pbar.set_postfix({
                'Processed': processed_count,
                'Errors': error_count,
                'Retries': retry_count,
                'ETA': format_time(eta)
            })
        
        return success, error

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(process_task, task): task 
                for task in tasks
            }
            
            while future_to_task and not stop_processing:
                done, not_done = concurrent.futures.wait(
                    future_to_task, 
                    timeout=1,
                    return_when=concurrent.futures.FIRST_COMPLETED
                )
                
                for future in done:
                    task = future_to_task.pop(future)
                    success, error = future.result()
                    
                    if error == "Requeued for retry":
                        new_future = executor.submit(process_task, task)
                        future_to_task[new_future] = task
                
                if stop_processing:
                    for future in not_done:
                        future.cancel()
                    break
    
    finally:
        pbar.close()
        
        total_time = time.time() - start_time
        completion_rate = processed_count / total_files * 100 if total_files > 0 else 0
        
        logger.info("\nProcessing completed!")
        logger.info(f"Final statistics:")
        logger.info(f"- Total time: {format_time(total_time)}")
        logger.info(f"- Processed: {processed_count} files")
        logger.info(f"- Errors: {error_count} files")
        logger.info(f"- Total retries: {retry_count}")
        logger.info(f"- Processing rate: {processed_count / total_time:.2f} files/second")
        logger.info(f"- Completion rate: {completion_rate:.2f}%")
        
        if stop_processing:
            logger.info("Script was interrupted by user")
            print("\nScript interrupted. Partial results have been saved.")
        else:
            print("\nEmbedding generation completed!")
        print(f"Total time: {format_time(total_time)}")
        print(f"Check resume_sfr_embedding.log for detailed processing information")

if __name__ == "__main__":
    input_dir = "data/resume/format_txt"
    output_dir = "data/resume/sfr_embedding"
    
    logger.info(f"Starting resume embedding process")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    create_embeddings(input_dir, output_dir, max_workers=400)