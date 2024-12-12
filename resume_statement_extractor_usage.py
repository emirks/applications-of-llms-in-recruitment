import os
import logging
import time
from datetime import datetime
import concurrent.futures
from dataclasses import dataclass
import threading
from tqdm.auto import tqdm
from resume_statement_extractor.resume_statement_extractor_local import StatementExtractor
import backoff
from typing import List, Tuple
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExtractionTask:
    filename: str
    category: str
    input_path: str
    retries: int = 0
    max_retries: int = 3

def get_unprocessed_resumes(base_path: str, categories: List[str]) -> List[ExtractionTask]:
    tasks = []
    for category in categories:
        txt_dir = os.path.join(base_path, 'format_txt', category)
        statements_dir = os.path.join(base_path, 'statements', 'format_json', category)
        
        if not os.path.exists(txt_dir):
            continue
            
        processed_files = {os.path.splitext(f)[0] for f in os.listdir(statements_dir)} if os.path.exists(statements_dir) else set()
        
        for txt_file in os.listdir(txt_dir):
            if txt_file.endswith('.txt'):
                file_id = os.path.splitext(txt_file)[0]
                if file_id not in processed_files:
                    tasks.append(ExtractionTask(
                        filename=file_id,
                        category=category,
                        input_path=os.path.join(txt_dir, txt_file)
                    ))
    
    random.shuffle(tasks)
    logger.info(f"Found {len(tasks)} unprocessed resumes")
    return tasks

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def process_resume_with_retry(extractor: StatementExtractor, task: ExtractionTask) -> Tuple[bool, str]:
    try:
        statements = extractor.extract_statements(task.input_path)
        extractor.save_statements(statements, task.category, task.filename)
        return True, ""
    except Exception as e:
        return False, str(e)

def process_resumes(base_path: str, categories: List[str], max_workers: int = 4):
    start_time = time.time()
    tasks = get_unprocessed_resumes(base_path, categories)
    if not tasks:
        logger.info("No new resumes to process")
        return

    processed_count = error_count = retry_count = 0
    pbar = tqdm(total=len(tasks), desc="Extracting Statements")
    counter_lock = threading.Lock()
    thread_local = threading.local()
    
    def get_extractor():
        if not hasattr(thread_local, "extractor"):
            thread_local.extractor = StatementExtractor()
        return thread_local.extractor

    def process_task(task: ExtractionTask) -> Tuple[bool, str]:
        nonlocal processed_count, error_count, retry_count
        success, error = process_resume_with_retry(get_extractor(), task)
        
        with counter_lock:
            if success:
                processed_count += 1
            else:
                if task.retries < task.max_retries:
                    task.retries += 1
                    retry_count += 1
                    return False, "Requeued for retry"
                error_count += 1
            pbar.update(1)
            pbar.set_postfix({'Processed': processed_count, 'Errors': error_count})
        return success, error

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_task, task): task for task in tasks}
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                success, error = future.result()
                if error == "Requeued for retry":
                    futures[executor.submit(process_task, task)] = task
    finally:
        pbar.close()
        logger.info(f"Processed: {processed_count} files, Errors: {error_count}, Retries: {retry_count}")

if __name__ == "__main__":
    base_path = "matcher_dataset/resume"
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
    
    process_resumes(base_path, categories, max_workers=4)
