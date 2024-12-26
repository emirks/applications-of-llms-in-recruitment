import os
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
from statement_pair_labeler.statement_pair_labeler import StatementPairLabeler
from statement_pair_labeler.labelers.gpt4o_labeler import LabelingError
from statement_pair_labeler.response_model import LabelDetails, LabeledPair
import glob
import re

# Rate limiting setup
max_concurrent_requests = 15
api_semaphore = threading.Semaphore(max_concurrent_requests)
requests_per_minute = 300
rate_limit_delay = 60.0 / requests_per_minute
last_request_time = threading.local()

# Thread-safe file writing lock
file_lock = threading.Lock()

def wait_for_rate_limit():
    current_time = time.time()
    if hasattr(last_request_time, 'time'):
        elapsed = current_time - last_request_time.time
        if elapsed < rate_limit_delay:
            time.sleep(rate_limit_delay - elapsed)
    last_request_time.time = time.time()

def save_pair(output_dir: str, pair: Dict[str, Any], pair_index: int):
    """Save a single pair to its own file with verification"""
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a filename using the pair index
    filename = f"pair_{pair_index:06d}.json"
    output_file = os.path.join(output_dir, filename)
    
    # Save the pair to its own file
    with file_lock:
        try:
            with open(output_file, 'w') as f:
                json.dump(pair, f, indent=2)
            
            # Verify the file was saved correctly
            try:
                with open(output_file, 'r') as f:
                    saved_pair = json.load(f)
                if saved_pair != pair:
                    raise ValueError(f"Verification failed for pair {pair_index}")
            except Exception as e:
                print(f"Error verifying saved pair {pair_index}: {str(e)}")
                # Try to save again
                with open(output_file, 'w') as f:
                    json.dump(pair, f, indent=2)
                
        except Exception as e:
            print(f"Error saving pair {pair_index}: {str(e)}")
            raise

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def label_pair_with_retry(labeler: StatementPairLabeler, pair: Dict[str, Any], 
                         output_file: str, is_first: bool) -> Dict[str, Any]:
    with api_semaphore:
        wait_for_rate_limit()
        label = labeler.labeler.label_pair(
            pair['jd_statement'],
            pair['skill_statement'],
            pair['metadata']
        )
        labeled_pair = {**pair, 'label': label}
        save_pair(output_file, labeled_pair, is_first)
        return labeled_pair

def process_batch(labeler: StatementPairLabeler, pairs: List[Dict[str, Any]], 
                 output_dir: str, start_idx: int) -> List[Dict[str, Any]]:
    """Process and save a batch of pairs"""
    try:
        with api_semaphore:
            wait_for_rate_limit()
            
            # Get batch labels with details
            label_results = labeler.labeler.label_pairs_batch(pairs)
            
            # Create labeled pairs
            labeled_pairs = []
            for idx, (pair, label_result) in enumerate(zip(pairs, label_results)):
                labeled_pair = {
                    **pair,
                    'label_details': {
                        'score': label_result['score'],
                        'category': label_result['category'],
                        'explanation': label_result['explanation']
                    }
                }
                save_pair(output_dir, labeled_pair, start_idx + idx)
                labeled_pairs.append(labeled_pair)
                
            return labeled_pairs
            
    except Exception as e:
        print(f"\nError processing batch starting at index {start_idx}:")
        print(str(e))
        raise

def load_existing_pairs(output_file: str) -> tuple[set[str], dict]:
    """Load already labeled pairs and return their IDs and data"""
    try:
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                data = json.load(f)
                # Create set of labeled pair IDs using jd_statement + skill_statement as key
                labeled_ids = set()
                for pair in data.get('pairs', []):
                    # Check if pair has required fields
                    if all(k in pair for k in ['jd_statement', 'skill_statement', 'label_details']):
                        # Check if label_details has required fields
                        if all(k in pair['label_details'] for k in ['score', 'category', 'explanation']):
                            pair_id = f"{pair['jd_statement']}|||{pair['skill_statement']}"
                            labeled_ids.add(pair_id)
                        else:
                            print(f"Warning: Incomplete label_details found in pair: {pair['jd_statement'][:50]}...")
                    else:
                        print(f"Warning: Incomplete pair data found in output file")
                
                print(f"Found {len(labeled_ids)} valid labeled pairs in existing file")
                return labeled_ids, data
    except json.JSONDecodeError as e:
        print(f"Warning: Error reading existing file {output_file}: {e}")
    except Exception as e:
        print(f"Warning: Unexpected error reading {output_file}: {e}")
    
    return set(), {
        'total_pairs': 0,
        'sampling_params': {},
        'labeling_info': {
            'model': 'gpt-4o-mini',
            'labeling_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'qualification_categories': {
                'NotRelevant': '0.0-0.1',
                'NotYetQualified': '0.1-0.5',
                'NearlyQualified': '0.5-0.8',
                'Qualified': '0.8-1.0'
            }
        },
        'pairs': []
    }

def main():
    input_file = "matcher_dataset/statement_pairs/statement_pairs_random.json"
    output_dir = "matcher_dataset/statement_pairs_labeled"
    batch_size = 12
    
    # Initialize labeler
    labeler = StatementPairLabeler('gpt4o_labeler')
    
    # Load pairs
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    total_input_pairs = len(data['pairs'])
    print(f"\nTotal pairs in input file: {total_input_pairs:,}")
    
    # Check existing labeled pairs
    existing_files = glob.glob(os.path.join(output_dir, "pair_*.json"))
    print(f"Previously labeled pairs: {len(existing_files):,}")
    
    # Get the highest existing index
    start_index = 0
    if existing_files:
        indices = [int(re.search(r'pair_(\d+)\.json', f).group(1)) for f in existing_files]
        start_index = max(indices) + 1
    
    # Filter pairs to start from the next unlabeled pair
    pairs = data['pairs'][start_index:]
    
    remaining_pairs = len(pairs)
    print(f"Remaining pairs to label: {remaining_pairs:,}")
    
    if not pairs:
        print("All pairs have already been labeled!")
        return
    
    # Create batches
    batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]
    total_batches = len(batches)
    
    print(f"\nProcessing in {total_batches} batches of {batch_size}...")
    
    failed_pairs = []
    
    with ThreadPoolExecutor(max_workers=max_concurrent_requests) as executor:
        future_to_batch = {
            executor.submit(
                process_batch, 
                labeler, 
                batch, 
                output_dir,
                start_index + (idx * batch_size)
            ): (idx, batch) 
            for idx, batch in enumerate(batches)
        }
        
        for future in tqdm(as_completed(future_to_batch), total=total_batches, 
                         desc="Processing batches"):
            try:
                future.result()
            except Exception as e:
                idx, batch = future_to_batch[future]
                batch_start_idx = start_index + (idx * batch_size)
                print(f"\nError processing batch starting at index {batch_start_idx}: {e}")
                failed_pairs.extend([(i, p) for i, p in enumerate(batch, batch_start_idx)])
    
    # Save failed pairs info
    if failed_pairs:
        failed_file = os.path.join(output_dir, "failed_pairs.json")
        with open(failed_file, 'w') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'failed_pairs': [{'index': idx, 'pair': pair} for idx, pair in failed_pairs]
            }, f, indent=2)
        print(f"\nWARNING: {len(failed_pairs)} pairs failed. See {failed_file} for details")
    
    final_files = glob.glob(os.path.join(output_dir, "pair_*.json"))
    print(f"\nLabeling complete! {len(final_files):,} pairs saved to {output_dir}")

if __name__ == "__main__":
    main() 