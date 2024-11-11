from resume_statement_extractor.resume_statement_extractor_local import StatementExtractor
import os
import json

# Base paths
base_path = "data/resume/format_json"
output_base_path = "data/resume/statements/format_json"

# Initialize extractor
resume_statement_extractor = StatementExtractor()

# Create output directory if it doesn't exist
os.makedirs(output_base_path, exist_ok=True)

# Process each JSON file
for json_file in os.listdir(base_path):
    if json_file.endswith('.json'):
        json_path = os.path.join(base_path, json_file)
        print(f"\nProcessing: {json_file}")
        
        try:
            # Read and parse JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                parsed_json = json.load(f)
            
            # Extract statements
            statements = resume_statement_extractor.extract_statements(file_path="", parsed_json=parsed_json)
            
            if statements:
                # Save to output directory
                output_file = os.path.join(output_base_path, json_file)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(statements, f, indent=2)
                print(f"Saved statements to: {output_file}")
                
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
