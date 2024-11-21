from resume_parser.resume_parser_local import ResumeParser
import os
from collections import defaultdict

# Base paths
base_path = "data/resume/format_pdf"
output_base_path = "data/resume/format_json"
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

# Initialize parser
resume_parser = ResumeParser()

# Create a dictionary to track unprocessed files for each category
unprocessed_files = defaultdict(list)

# Initialize unprocessed files for each category
for category in categories:
    category_path = os.path.join(base_path, category)
    output_category_path = os.path.join(output_base_path, category)
    
    # Skip if category directory doesn't exist
    if not os.path.exists(category_path):
        print(f"Skipping {category} - directory not found")
        continue
    
    # Create output category directory if it doesn't exist
    os.makedirs(output_category_path, exist_ok=True)
    
    # Get list of unprocessed PDFs
    for resume_file in os.listdir(category_path):
        if resume_file.endswith('.pdf'):
            output_json_path = os.path.join(output_category_path, 
                                          f"{os.path.splitext(resume_file)[0]}.json")
            if not os.path.exists(output_json_path):
                unprocessed_files[category].append(resume_file)

print("Unprocessed files:")
for category, files in unprocessed_files.items():
    print(f"{category}: {len(files)}")

# Process files in round-robin fashion
while any(unprocessed_files.values()):  # Continue while any category has unprocessed files
    for category in categories:
        if unprocessed_files[category]:  # If category has unprocessed files
            # Get next file to process
            resume_file = unprocessed_files[category][0]
            
            print(f"\nProcessing from category {category}: {resume_file}")
            
            resume_path = os.path.join(base_path, category, resume_file)
            user_id = f"{category}_{os.path.splitext(resume_file)[0]}"
            
            try:
                # Process the file
                resume_parser.process_file(resume_path, user_id, category, save_json=True)
                # Remove processed file from the list
                unprocessed_files[category].pop(0)
                print(f"Successfully processed: {resume_file}")
            except Exception as e:
                print(f"Error processing {resume_file}: {str(e)}")
                # Optionally, you might want to remove failed files to avoid infinite loops
                unprocessed_files[category].pop(0)
                print(f"Removed failed file from queue: {resume_file}")

print("\nAll resumes have been processed!")