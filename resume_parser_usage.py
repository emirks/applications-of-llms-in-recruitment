import os
import logging
import time
from datetime import datetime, timedelta
import concurrent.futures
from dataclasses import dataclass
import threading
from tqdm.auto import tqdm
from resume_parser.resume_parser_local import ResumeParser
import backoff
from typing import List, Tuple
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ParsingTask:
    filename: str
    category: str
    input_path: str
    output_path: str
    retries: int = 0
    max_retries: int = 3

def get_unprocessed_resumes(base_path: str, output_base_path: str, categories: List[str]) -> List[ParsingTask]:
    tasks = []
    for category in categories:
        txt_dir = os.path.join(base_path, 'format_txt', category)
        json_dir = os.path.join(output_base_path, 'format_json', category)
        
        if not os.path.exists(txt_dir):
            continue
            
        processed_files = {os.path.splitext(f)[0] for f in os.listdir(json_dir)} if os.path.exists(json_dir) else set()
        
        for txt_file in os.listdir(txt_dir):
            if txt_file.endswith('.txt'):
                file_id = os.path.splitext(txt_file)[0]
                if file_id not in processed_files:
                    tasks.append(ParsingTask(
                        filename=txt_file,
                        category=category,
                        input_path=os.path.join(base_path, 'format_pdf', category, f"{file_id}.pdf"),
                        output_path=os.path.join(json_dir, f"{file_id}.json")
                    ))
    
    random.shuffle(tasks)
    logger.info(f"Found {len(tasks)} unprocessed resumes")
    return tasks

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def process_resume_with_retry(resume_parser: ResumeParser, task: ParsingTask) -> Tuple[bool, str]:
    try:
        user_id = f"{task.category}_{os.path.splitext(task.filename)[0]}"
        resume_parser.process_file(task.input_path, user_id, task.category, save_json=True)
        return True, ""
    except Exception as e:
        return False, str(e)

def process_resumes(base_path: str, output_base_path: str, categories: List[str], max_workers: int = 4):
    start_time = time.time()
    tasks = get_unprocessed_resumes(base_path, output_base_path, categories)
    if not tasks:
        logger.info("No new resumes to process")
        return

    processed_count = error_count = retry_count = 0
    pbar = tqdm(total=len(tasks), desc="Processing Resumes")
    counter_lock = threading.Lock()
    thread_local = threading.local()
    
    def get_parser():
        if not hasattr(thread_local, "parser"):
            thread_local.parser = ResumeParser()
        return thread_local.parser

    def process_task(task: ParsingTask) -> Tuple[bool, str]:
        nonlocal processed_count, error_count, retry_count
        success, error = process_resume_with_retry(get_parser(), task)
        
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
    output_base_path = "matcher_dataset/resume"
    categories = ["engineering", "customer_service", "education_and_training", 
                 "finance_and_accounting", "healthcare_and_medicine", "human_resources",
                 "information_technology", "legal_and_compliance", 
                 "logistics_and_supply_chain", "marketing_and_sales"]
    
    process_resumes(base_path, output_base_path, categories, max_workers=4)