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
import concurrent.futures
from typing import List, Tuple, Dict
from dataclasses import dataclass
from queue import Queue
import threading
import signal
import sys
import backoff  # You'll need to pip install backoff
from requests.exceptions import RequestException

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('job_description_embedding.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class EmbeddingTask:
    filename: str
    input_path: str
    output_path: str
    text: str
    retries: int = 0
    max_retries: int = 3

def format_time(seconds):
    """Convert seconds to human readable time format"""
    return str(timedelta(seconds=int(seconds)))

def should_retry(e):
    """Determine if the error is retryable"""
    if isinstance(e, RequestException):
        # Retry on connection errors, timeouts, and 5xx errors
        return True
    if isinstance(e, ModelServiceError):
        # You might want to customize this based on your ModelServiceError types
        return True
    return False

@backoff.on_exception(
    backoff.expo,
    (RequestException, ModelServiceError),
    max_tries=3,
    giveup=lambda e: not should_retry(e)
)
def get_embedding_with_retry(model: SFREmbeddingEndpoint, text: str) -> np.ndarray:
    """Get embedding with exponential backoff retry"""
    return model.get_embedding(text)

def process_file(task: EmbeddingTask, embedding_model: SFREmbeddingEndpoint) -> Tuple[str, bool, str]:
    """Process a single file and return results"""
    try:
        embedding = get_embedding_with_retry(embedding_model, task.text)
        np.save(task.output_path, embedding)
        return task.filename, True, ""
    except Exception as e:
        error_msg = f"Failed after {task.max_retries} retries: {str(e)}"
        return task.filename, False, error_msg

def create_embedding_tasks(input_dir: str, output_dir: str) -> List[EmbeddingTask]:
    """Create list of embedding tasks from input files"""
    tasks = []
    for filename in os.listdir(input_dir):
        if not filename.endswith('.txt'):
            continue
            
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace('.txt', '.npy'))
        
        if os.path.exists(output_path):
            continue
            
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            if text:
                tasks.append(EmbeddingTask(
                    filename=filename,
                    input_path=input_path,
                    output_path=output_path,
                    text=text
                ))
        except Exception as e:
            logger.error(f"Error reading {filename}: {str(e)}")
            
    return tasks

stop_processing = False

def signal_handler(signum, frame):
    """Handle interrupt signal"""
    global stop_processing
    print("\nReceived interrupt signal. Stopping gracefully...")
    stop_processing = True

def create_embeddings(input_dir: str, output_dir: str, max_workers: int = 100):
    """Create embeddings using multiple workers"""
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    start_time = time.time()
    logger.info(f"Starting embedding generation process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create tasks
    tasks = create_embedding_tasks(input_dir, output_dir)
    total_files = len(tasks)
    random.shuffle(tasks)
    
    logger.info(f"Found {total_files} files to process")
    
    # Initialize counters
    processed_count = 0
    error_count = 0
    retry_count = 0
    
    # Create progress bar
    pbar = tqdm(total=total_files, desc="Processing Files")
    
    # Create thread-safe counters
    counter_lock = threading.Lock()
    
    def process_task(task: EmbeddingTask) -> Tuple[bool, str]:
        """Process single task with dedicated model instance"""
        if stop_processing:
            return False, "Interrupted by user"
            
        model = SFREmbeddingEndpoint()
        filename, success, error = process_file(task, model)
        
        nonlocal processed_count, error_count, retry_count
        with counter_lock:
            if success:
                processed_count += 1
            else:
                if task.retries < task.max_retries:
                    # Requeue the task
                    task.retries += 1
                    retry_count += 1
                    logger.warning(f"Retrying {filename} (attempt {task.retries}/{task.max_retries})")
                    return False, "Requeued for retry"
                else:
                    error_count += 1
                    logger.error(f"Error processing {filename}: {error}")
            
            # Update progress bar
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
        # Process files using thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Initial submission of tasks
            future_to_task = {
                executor.submit(process_task, task): task 
                for task in tasks
            }
            
            # Process tasks and handle retries
            while future_to_task and not stop_processing:
                done, not_done = concurrent.futures.wait(
                    future_to_task, 
                    timeout=1,
                    return_when=concurrent.futures.FIRST_COMPLETED
                )
                
                for future in done:
                    task = future_to_task.pop(future)
                    success, error = future.result()
                    
                    # If task needs retry, resubmit it
                    if error == "Requeued for retry":
                        new_future = executor.submit(process_task, task)
                        future_to_task[new_future] = task
                
                if stop_processing:
                    # Cancel pending tasks
                    for future in not_done:
                        future.cancel()
                    break
    
    finally:
        pbar.close()
        
        # Log final statistics
        total_time = time.time() - start_time
        completion_rate = processed_count / total_files * 100
        
        logger.info("\nProcessing completed!")
        logger.info(f"Final statistics:")
        logger.info(f"- Total time: {format_time(total_time)}")
        logger.info(f"- Processed: {processed_count} files")
        logger.info(f"- Errors: {error_count} files")
        logger.info(f"- Total retries: {retry_count}")
        logger.info(f"- Processing rate: {processed_count / total_time:.2f} files/second")
        logger.info(f"- Total completion rate: {completion_rate:.2f}%")
        
        if stop_processing:
            logger.info("Script was interrupted by user")
            print("\nScript interrupted. Partial results have been saved.")
        else:
            print("\nEmbedding generation completed!")
        print(f"Total time: {format_time(total_time)}")
        print(f"Check job_description_embedding.log for detailed processing information")

if __name__ == "__main__":
    input_dir = "data/job_descriptions/format_txt"
    output_dir = "data/job_descriptions/sfr_embedding"
    create_embeddings(input_dir, output_dir, max_workers=400)