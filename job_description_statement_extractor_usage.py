from job_description_statement_extractor.job_description_statement_extractor import StatementExtractor
import os
import json

# Base paths
base_path = r"data\\job_descriptions\\format_txt"
output_base_path = r"data\\job_descriptions\\statements\\format_json"

# Initialize extractor
statement_extractor = StatementExtractor()

# Create output directory if it doesn't exist
os.makedirs(output_base_path, exist_ok=True)

# Process each text file
for txt_file in os.listdir(base_path):
    if txt_file.endswith('.txt'):
        txt_path = os.path.join(base_path, txt_file)
        print(f"\nProcessing: {txt_file}")
        
        try:
            # Extract statements directly from text file
            statements = statement_extractor.extract_statements(file_path=txt_path)
            
            if statements:
                # Save statements as JSON
                output_json_file = os.path.join(output_base_path, f"{os.path.splitext(txt_file)[0]}.json")
                with open(output_json_file, 'w', encoding='utf-8') as f:
                    json.dump(statements, f, indent=2)
                print(f"Saved JSON statements to: {output_json_file}")
                
                # Save formatted output as text
                output_text_file = os.path.join(output_base_path.replace('format_json', 'format_txt'), txt_file)
                formatted_output = statement_extractor.format_output(statements)
                with open(output_text_file, 'w', encoding='utf-8') as f:
                    f.write(formatted_output)
                print(f"Saved formatted statements to: {output_text_file}")
                
        except Exception as e:
            print(f"Error processing {txt_file}: {str(e)}")
