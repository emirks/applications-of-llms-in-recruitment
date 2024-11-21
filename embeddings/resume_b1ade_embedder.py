import os
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from embeddings.embedding_models.b1ade_embed import B1adeEmbed
from resume_parser.resume_parser_local import ResumeParser
import random
import logging
import time
from datetime import datetime, timedelta
import concurrent.futures
import threading
import signal
import backoff
from typing import List, Tuple
from dataclasses import dataclass
import sys
from PIL import Image
import fitz  # PyMuPDF
import io
import pytesseract

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('resume_embedding.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class EmbeddingTask:
    filename: str
    category: str
    input_path: str
    output_path: str
    txt_path: str
    retries: int = 0
    max_retries: int = 3

# Add global lock for model initialization
model_init_lock = threading.Lock()
thread_local = threading.local()

# Add at the top level
stop_processing = False

# Add at top with other locks
parser_lock = threading.Lock()

def signal_handler(signum, frame):
    global stop_processing
    stop_processing = True
    print("\nGraceful shutdown initiated. Waiting for current tasks to complete...")

def get_models():
    """Get or initialize models with thread safety"""
    if not hasattr(thread_local, "models"):
        with model_init_lock:  # Use lock for model initialization
            # Double-check pattern to prevent race conditions
            if not hasattr(thread_local, "models"):
                logger.info("Initializing new model instance for thread")
                time.sleep(random.uniform(0.5, 2.0))  # Stagger initializations
                thread_local.models = {
                    'embedding': B1adeEmbed(),
                    'parser': ResumeParser()
                }
    return thread_local.models

@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=3,
    jitter=backoff.full_jitter,
    base=3,
    max_value=60
)
def process_resume_with_retry(task: EmbeddingTask) -> Tuple[bool, str]:
    try:
        models = get_models()
        
        # Extract text with visual processing
        logger.info(f"Starting text extraction for {task.filename}")
        text = extract_text_with_ocr(task.input_path)
        if not text or len(text.strip()) < 50:  # Minimum text length check
            return False, "Insufficient text extracted"
            
        # Save text file
        with open(task.txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        logger.info(f"Successfully extracted and saved text for {task.filename} ({len(text)} characters)")
            
        # Get and save embedding
        logger.info(f"Generating embedding for {task.filename}")
        embedding = models['embedding'].get_embedding(text)
        np.save(task.output_path, embedding)
        logger.info(f"Successfully generated and saved embedding for {task.filename}")
        
        return True, ""
    except Exception as e:
        return False, str(e)

def create_embedding_tasks(base_path: str, output_base_path: str, txt_base_path: str, categories: List[str]) -> List[EmbeddingTask]:
    tasks = []
    for category in categories:
        category_path = os.path.join(base_path, category)
        output_category_path = os.path.join(output_base_path, category)
        txt_category_path = os.path.join(txt_base_path, category)
        
        Path(output_category_path).mkdir(parents=True, exist_ok=True)
        Path(txt_category_path).mkdir(parents=True, exist_ok=True)
        
        for pdf_file in os.listdir(category_path):
            if not pdf_file.endswith('.pdf'):
                continue
                
            input_path = os.path.join(category_path, pdf_file)
            output_path = os.path.join(output_category_path, pdf_file.replace('.pdf', '.npy'))
            txt_path = os.path.join(txt_category_path, pdf_file.replace('.pdf', '.txt'))
            
            if os.path.exists(output_path) and os.path.exists(txt_path):
                continue
                
            tasks.append(EmbeddingTask(
                filename=pdf_file,
                category=category,
                input_path=input_path,
                output_path=output_path,
                txt_path=txt_path
            ))
    
    return tasks

def create_embeddings(input_base_dir: str, output_base_dir: str, txt_base_dir: str, max_workers: int = 5):
    try:
        if not os.path.exists(input_base_dir):
            raise ValueError(f"Input directory does not exist: {input_base_dir}")
            
        Path(output_base_dir).mkdir(parents=True, exist_ok=True)
        Path(txt_base_dir).mkdir(parents=True, exist_ok=True)
        
        signal.signal(signal.SIGINT, signal_handler)
        start_time = time.time()
        logger.info(f"Starting resume embedding at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Pre-initialize models in main thread
        logger.info("Pre-initializing models in main thread")
        try:
            main_models = {
                'embedding': B1adeEmbed(),
                'parser': ResumeParser()
            }
            thread_local.models = main_models
        except Exception as e:
            logger.error(f"Failed to initialize models: {str(e)}")
            raise
        
        # Get categories and create tasks
        categories = [d for d in os.listdir(input_base_dir) 
                     if os.path.isdir(os.path.join(input_base_dir, d))]
        
        tasks = create_embedding_tasks(input_base_dir, output_base_dir, txt_base_dir, categories)
        total_files = len(tasks)
        random.shuffle(tasks)
        logger.info(f"Found {total_files} files to process")
        
        # Initialize counters
        processed_count = 0
        error_count = 0
        retry_count = 0
        
        # Create progress bar
        pbar = tqdm(total=total_files, desc="Processing Resumes")
        counter_lock = threading.Lock()
        
        def process_task(task: EmbeddingTask) -> Tuple[bool, str]:
            if stop_processing:
                return False, "Interrupted by user"
            
            nonlocal processed_count, error_count, retry_count
            try:
                success, error = process_resume_with_retry(task)
                
                with counter_lock:
                    if success:
                        processed_count += 1
                        logger.info(f"Successfully completed processing {task.filename}")
                    else:
                        if task.retries < task.max_retries:
                            task.retries += 1
                            retry_count += 1
                            logger.warning(f"Retrying {task.filename} (attempt {task.retries}/{task.max_retries})")
                            return False, "Requeued for retry"
                        else:
                            error_count += 1
                            logger.error(f"Error processing {task.filename}: {error}")
                    
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
            except Exception as e:
                logger.error(f"Error in process_task for {task.filename}: {str(e)}", exc_info=True)
                with counter_lock:
                    error_count += 1
                    pbar.update(1)
                return False, str(e)
        
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
                        try:
                            success, error = future.result()
                            if error == "Requeued for retry":
                                new_future = executor.submit(process_task, task)
                                future_to_task[new_future] = task
                        except concurrent.futures.CancelledError:
                            logger.warning(f"Task cancelled for {task.filename}")
                        except Exception as e:
                            logger.error(f"Error processing {task.filename}: {str(e)}", exc_info=True)
                            error_count += 1
                    
                    if stop_processing:
                        for future in not_done:
                            future.cancel()
                        break
        finally:
            pbar.close()
            # Cleanup
            B1adeEmbed.free_memory()
            
            total_time = time.time() - start_time
            completion_rate = processed_count / total_files * 100 if total_files > 0 else 0
            skipped = total_files - (processed_count + error_count)
            
            logger.info("\nProcessing completed!")
            logger.info(f"Final statistics:")
            logger.info(f"- Total time: {format_time(total_time)}")
            logger.info(f"- Processed: {processed_count} files")
            logger.info(f"- Errors: {error_count} files")
            logger.info(f"- Skipped: {skipped} files")
            logger.info(f"- Total retries: {retry_count}")
            logger.info(f"- Processing rate: {processed_count / total_time:.2f} files/second")
            logger.info(f"- Completion rate: {completion_rate:.2f}%")
            
            if stop_processing:
                logger.info("Script was interrupted by user")
                print("\nScript interrupted. Partial results have been saved.")
            else:
                print("\nEmbedding generation completed!")
            print(f"Total time: {format_time(total_time)}")
            print(f"Check resume_embedding.log for detailed processing information")
    except Exception as e:
        logger.error(f"Fatal error occurred: {str(e)}", exc_info=True)
        print(f"\nFatal error occurred: {str(e)}")
        print("Check resume_embedding.log for detailed error information")
        sys.exit(1)
    finally:
        logger.info(f"Script execution ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nScript execution completed. Check resume_embedding.log for details.")

def format_time(seconds):
    """Convert seconds to human readable time format"""
    return str(timedelta(seconds=int(seconds)))

def extract_text_with_ocr(pdf_path: str) -> str:
    """Extract text using Tesseract OCR"""
    try:
        doc = fitz.open(pdf_path)
        images = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            images.append(img_array)
        
        doc.close()
            
        # Use OCR with preprocessing
        from resume_parser.parsers.tesseract_parser import TesseractParser
        parser = TesseractParser()
        
        # Preprocess images
        preprocessed_images = parser.preprocess_images(images)
        
        # Get OCR text
        text = parser.parse_text(pdf_path, preprocessed_images)
        
        return text.strip()
        
    except Exception as e:
        logger.error(f"Error in text extraction: {str(e)}")
        return ""

if __name__ == "__main__":
    try:
        input_dir = "data/resume/format_pdf"
        output_dir = "data/resume/b1ade_embedding"
        txt_dir = "data/resume/format_txt"
        
        logger.info(f"Starting resume embedding process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Text directory: {txt_dir}")
        
        create_embeddings(input_dir, output_dir, txt_dir, max_workers=3)
        
    except KeyboardInterrupt:
        logger.error("Script interrupted by user")
        print("\nScript interrupted by user. Check resume_embedding.log for details.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error occurred: {str(e)}", exc_info=True)
        print(f"\nFatal error occurred: {str(e)}")
        print("Check resume_embedding.log for detailed error information")
        sys.exit(1)
    finally:
        logger.info(f"Script execution ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nScript execution completed. Check resume_embedding.log for details.")