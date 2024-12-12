from job_description_statement_extractor.job_description_statement_extractor import StatementExtractor
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List
import threading
import time
from tqdm import tqdm
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential
import random

# Semaphore to control concurrent API requests
# GPT-4o-mini allows 500 requests per minute
max_concurrent_requests = 15
api_semaphore = threading.Semaphore(max_concurrent_requests)

# Rate limiting parameters
requests_per_minute = 300  # GPT-4o-mini's RPM limit
rate_limit_delay = 60.0 / requests_per_minute
last_request_time = threading.local()

def wait_for_rate_limit():
    """Ensure we don't exceed the rate limit"""
    current_time = time.time()
    if hasattr(last_request_time, 'time'):
        elapsed = current_time - last_request_time.time
        if elapsed < rate_limit_delay:
            time.sleep(rate_limit_delay - elapsed)
    last_request_time.time = time.time()

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def extract_with_retry(statement_extractor: StatementExtractor, file_path: str) -> Tuple[dict, str]:
    """Extract statements with retry logic"""
    statements = statement_extractor.extract_statements(file_path=file_path)
    if statements:
        formatted_output = statement_extractor.format_output(statements)
        return statements, formatted_output
    raise Exception("Failed to extract statements")

def process_single_file(txt_path: str, statement_extractor: StatementExtractor) -> Tuple[str, dict, str]:
    try:
        # Acquire semaphore and handle rate limiting
        with api_semaphore:
            wait_for_rate_limit()
            # Extract statements with retry logic
            try:
                statements, formatted_output = extract_with_retry(statement_extractor, txt_path)
                return txt_path, statements, formatted_output
            except tenacity.RetryError as e:
                print(f"\nFailed after 3 retries for {os.path.basename(txt_path)}: {str(e)}")
            except Exception as e:
                print(f"\nUnexpected error processing {os.path.basename(txt_path)}: {str(e)}")
                
    except Exception as e:
        print(f"\nError processing {os.path.basename(txt_path)}: {str(e)}")
    
    return txt_path, None, None

def save_outputs(txt_path: str, statements: dict, formatted_output: str, output_base_path: str):
    if statements:
        try:
            # Save statements as JSON
            txt_filename = os.path.basename(txt_path)
            output_json_file = os.path.join(output_base_path, f"{os.path.splitext(txt_filename)[0]}.json")
            with open(output_json_file, 'w', encoding='utf-8') as f:
                json.dump(statements, f, indent=2)
            print(f"Saved JSON statements to: {output_json_file}")
            
            # Save formatted output as text
            output_text_file = os.path.join(output_base_path.replace('format_json', 'format_txt'), txt_filename)
            with open(output_text_file, 'w', encoding='utf-8') as f:
                f.write(formatted_output)
            print(f"Saved formatted statements to: {output_text_file}")
        except Exception as e:
            print(f"Error saving outputs for {txt_filename}: {str(e)}")

def should_continue():
    return not os.path.exists("stop.txt")

def get_unprocessed_files(input_files: List[str], output_base_path: str) -> List[str]:
    """Return list of files that haven't been processed yet"""
    # Get list of existing output files
    json_output_dir = output_base_path
    txt_output_dir = output_base_path.replace('format_json', 'format_txt')
    
    processed_files = set()
    
    # Check JSON outputs with more detailed logging
    if os.path.exists(json_output_dir):
        json_files = [f for f in os.listdir(json_output_dir) if f.endswith('.json')]
        processed_files.update(os.path.splitext(f)[0] for f in json_files)
        print(f"Found {len(json_files)} existing JSON files")
    
    # Check text outputs with more detailed logging
    if os.path.exists(txt_output_dir):
        txt_files = [f for f in os.listdir(txt_output_dir) if f.endswith('.txt')]
        processed_files.update(os.path.splitext(f)[0] for f in txt_files)
        print(f"Found {len(txt_files)} existing TXT files")
    
    # Debug print some sample processed files
    sample_processed = list(processed_files)[:5]
    if sample_processed:
        print("Sample processed files:", sample_processed)
    
    # Get sample input files
    sample_input = [os.path.splitext(os.path.basename(f))[0] for f in input_files[:5]]
    print("Sample input files:", sample_input)
    
    # Filter out already processed files with detailed logging
    unprocessed_files = []
    for input_file in input_files:
        file_id = os.path.splitext(os.path.basename(input_file))[0]
        if file_id not in processed_files:
            unprocessed_files.append(input_file)
    
    total_files = len(input_files)
    processed_count = total_files - len(unprocessed_files)
    
    print(f"\nTotal input files: {total_files}")
    print(f"Previously processed: {processed_count}")
    print(f"Remaining to process: {len(unprocessed_files)}")
    
    return unprocessed_files

def main():
    # Base paths
    base_path = r"matcher_dataset\\job_descriptions\\format_txt"
    output_base_path = r"matcher_dataset\\job_descriptions\\statements\\format_json"

    print(f"\nChecking input directory: {base_path}")
    print(f"Checking output directory: {output_base_path}")
    
    # Create output directories
    os.makedirs(output_base_path, exist_ok=True)
    os.makedirs(output_base_path.replace('format_json', 'format_txt'), exist_ok=True)

    # Initialize extractor
    statement_extractor = StatementExtractor()

    # Get list of all text files with size check
    all_txt_files = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('.txt')]
    print(f"\nFound {len(all_txt_files)} total input files")
    
    # Shuffle all files once
    random.shuffle(all_txt_files)
    print("Files shuffled randomly")
    
    # Filter out already processed files
    txt_files = get_unprocessed_files(all_txt_files, output_base_path)
    
    if not txt_files:
        print("\nNo new files to process")
        return
        
    # Process files in parallel with rate limiting
    max_workers = min(max_concurrent_requests * 2, len(txt_files))  # More workers than concurrent requests to handle I/O
    print(f"Processing {len(txt_files)} files with {max_workers} workers...")
    
    # Initialize progress bar
    pbar = tqdm(total=len(txt_files), desc="Processing files", 
               unit="file", ncols=100, 
               bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    success_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_file, txt_path, statement_extractor): txt_path 
            for txt_path in txt_files
        }
        
        # Process completed tasks
        try:
            for future in as_completed(future_to_file):
                if not should_continue():
                    print("\nStop file detected. Gracefully shutting down...")
                    executor.shutdown(wait=True, cancel_futures=True)
                    break
                    
                txt_path, statements, formatted_output = future.result()
                if statements:
                    save_outputs(txt_path, statements, formatted_output, output_base_path)
                    success_count += 1
                else:
                    error_count += 1
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'success': success_count,
                    'errors': error_count
                })
                
        except KeyboardInterrupt:
            print("\nInterrupt received. Gracefully shutting down...")
            executor.shutdown(wait=True, cancel_futures=True)
        
        finally:
            pbar.close()
            
    # Print final statistics
    print(f"\nProcessing completed:")
    print(f"Successfully processed: {success_count} files")
    print(f"Errors encountered: {error_count} files")
    print(f"Total files processed: {success_count + error_count}")

if __name__ == "__main__":
    main()
