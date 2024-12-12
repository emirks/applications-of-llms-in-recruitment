from job_description_statement_extractor.job_description_statement_extractor import StatementExtractor
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple
import threading
import time

# Semaphore to control concurrent API requests
# Adjust max_concurrent_requests based on your API limits
max_concurrent_requests = 10
api_semaphore = threading.Semaphore(max_concurrent_requests)

# Rate limiting parameters
requests_per_minute = 50  # Adjust based on your API limits
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

def process_single_file(txt_path: str, statement_extractor: StatementExtractor) -> Tuple[str, dict, str]:
    try:
        # Acquire semaphore and handle rate limiting
        with api_semaphore:
            wait_for_rate_limit()
            # Extract statements directly from text file
            statements = statement_extractor.extract_statements(file_path=txt_path)
            
            if statements:
                # Generate formatted output
                formatted_output = statement_extractor.format_output(statements)
                return txt_path, statements, formatted_output
                
    except Exception as e:
        print(f"Error processing {os.path.basename(txt_path)}: {str(e)}")
    
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

def main():
    # Base paths
    base_path = r"matcher_dataset\\job_descriptions\\format_txt"
    output_base_path = r"matcher_dataset\\job_descriptions\\statements\\format_json"

    # Create output directories
    os.makedirs(output_base_path, exist_ok=True)
    os.makedirs(output_base_path.replace('format_json', 'format_txt'), exist_ok=True)

    # Initialize extractor
    statement_extractor = StatementExtractor()

    # Get list of text files
    txt_files = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('.txt')]
    
    # Process files in parallel with rate limiting
    max_workers = min(max_concurrent_requests * 2, len(txt_files))  # More workers than concurrent requests to handle I/O
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_file, txt_path, statement_extractor): txt_path 
            for txt_path in txt_files
        }
        
        # Process completed tasks
        for future in as_completed(future_to_file):
            txt_path, statements, formatted_output = future.result()
            if statements:
                save_outputs(txt_path, statements, formatted_output, output_base_path)

if __name__ == "__main__":
    main()
