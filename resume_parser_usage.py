import os
import logging
import time
from datetime import datetime, timedelta
import concurrent.futures
from pathlib import Path
from dataclasses import dataclass
import threading
from tqdm.auto import tqdm
from resume_parser.resume_parser_local import ResumeParser
import signal
import backoff
from typing import List, Tuple, Dict
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('resume_parsing.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ParsingTask:
    filename: str
    category: str
    input_path: str
    output_path: str
    retries: int = 0
    max_retries: int = 3

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def create_parsing_tasks(base_path: str, output_base_path: str, categories: List[str]) -> List[ParsingTask]:
    """Create list of parsing tasks from input files"""
    tasks = []
    
    for category in categories:
        category_path = os.path.join(base_path, category)
        output_category_path = os.path.join(output_base_path, category)
        
        if not os.path.exists(category_path):
            logger.warning(f"Skipping {category} - directory not found")
            continue
            
        Path(output_category_path).mkdir(parents=True, exist_ok=True)
        
        for resume_file in os.listdir(category_path):
            if resume_file.endswith('.pdf'):
                output_json_path = os.path.join(
                    output_category_path, 
                    f"{os.path.splitext(resume_file)[0]}.json"
                )
                
                if os.path.exists(output_json_path):
                    continue
                    
                input_path = os.path.join(category_path, resume_file)
                tasks.append(ParsingTask(
                    filename=resume_file,
                    category=category,
                    input_path=input_path,
                    output_path=output_json_path
                ))
    
    return tasks

stop_processing = False

def signal_handler(signum, frame):
    global stop_processing
    print("\nReceived interrupt signal. Stopping gracefully...")
    stop_processing = True

@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=3,
    jitter=backoff.full_jitter,
    base=3,
    max_value=60
)
def process_resume_with_retry(resume_parser: ResumeParser, task: ParsingTask) -> Tuple[bool, str]:
    """Process single resume with retry logic"""
    try:
        user_id = f"{task.category}_{os.path.splitext(task.filename)[0]}"
        resume_parser.process_file(task.input_path, user_id, task.category, save_json=True)
        return True, ""
    except Exception as e:
        return False, str(e)

def process_resumes(base_path: str, output_base_path: str, categories: List[str], max_workers: int = 10):
    """Process resumes using multiple workers"""
    signal.signal(signal.SIGINT, signal_handler)
    
    start_time = time.time()
    logger.info(f"Starting resume parsing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create tasks
    tasks = create_parsing_tasks(base_path, output_base_path, categories)
    total_files = len(tasks)
    logger.info(f"Found {total_files} files to process")
    
    # Initialize counters
    processed_count = 0
    error_count = 0
    retry_count = 0
    
    # Create progress bar
    pbar = tqdm(total=total_files, desc="Processing Resumes")
    
    # Create thread-safe counter
    counter_lock = threading.Lock()
    
    # Thread-local storage for ResumeParser instances
    thread_local = threading.local()
    
    def get_parser():
        if not hasattr(thread_local, "parser"):
            time.sleep(random.uniform(0.5, 2.0))
            thread_local.parser = ResumeParser()
        return thread_local.parser
    
    def process_task(task: ParsingTask) -> Tuple[bool, str]:
        if stop_processing:
            return False, "Interrupted by user"
            
        # Get thread-local parser instance
        resume_parser = get_parser()
        success, error = process_resume_with_retry(resume_parser, task)
        
        nonlocal processed_count, error_count, retry_count
        with counter_lock:
            if success:
                processed_count += 1
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
            print("\nResume parsing completed!")
        print(f"Total time: {format_time(total_time)}")
        print(f"Check resume_parsing.log for detailed processing information")

if __name__ == "__main__":
    # Base paths
    base_path = "data/resume/format_pdf"
    output_base_path = "data/resume/format_json"
    
    # Categories list from original script
    categories = [
        "engineering",
        "customer_service",
        "education_and_training",
        "finance_and_accounting",
        "healthcare_and_medicine",
        "human_resources",
        "information_technology",
        "legal_and_compliance",
        "logistics_and_supply_chain",
        "marketing_and_sales"
    ]
    
    process_resumes(base_path, output_base_path, categories, max_workers=4)