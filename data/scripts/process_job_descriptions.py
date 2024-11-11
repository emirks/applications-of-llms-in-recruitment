import json
import os
from pathlib import Path

def create_job_description_files(input_file: str, output_dir: str, num_files: int = 1000):
    """
    Process job descriptions from a JSON file and create individual text files
    
    Args:
        input_file (str): Path to input JSON file with job descriptions
        output_dir (str): Directory to store output text files
        num_files (int): Number of files to process (default 1000)
    """
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if count >= num_files:
                break
                
            try:
                # Parse JSON line
                job_data = json.loads(line.strip())
                
                # Get job ID
                job_id = job_data['_id']['$oid']
                
                # Extract text content
                text_content = job_data.get('text', '')
                
                if text_content:
                    # Create output file path
                    output_file = os.path.join(output_dir, f"{job_id}.txt")
                    
                    # Write text content to file
                    with open(output_file, 'w', encoding='utf-8') as out_f:
                        out_f.write(text_content)
                    
                    count += 1
                    
                    if count % 100 == 0:
                        print(f"Processed {count} files...")
                        
            except json.JSONDecodeError:
                print(f"Error parsing JSON line")
                continue
            except Exception as e:
                print(f"Error processing job description: {e}")
                continue
    
    print(f"Successfully created {count} job description files")

if __name__ == "__main__":
    # Configure paths
    input_file = r"job_descriptions\\techmap-jobs-export-2022-10_ie.json"  # Your input JSON file
    output_dir = r"data\job_descriptions\\format_txt"       # Output directory for text files
    
    # Process job descriptions
    create_job_description_files(input_file, output_dir)
