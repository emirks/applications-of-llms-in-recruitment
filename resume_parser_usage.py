from resume_parser.resume_parser_local import ResumeParser
import os

# Base paths
base_path = "data/resumes/format_pdf"
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

# Process each category
for category in categories:
    category_path = os.path.join(base_path, category)
    
    # Skip if category directory doesn't exist
    if not os.path.exists(category_path):
        print(f"Skipping {category} - directory not found")
        continue
    
    print(f"\nProcessing category: {category}")
    
    # Process each resume in the category
    for resume_file in os.listdir(category_path):
        if resume_file.endswith('.pdf'):
            resume_path = os.path.join(category_path, resume_file)
            user_id = f"{category}_{os.path.splitext(resume_file)[0]}"
            
            print(f"Processing: {resume_file}")
            try:
                resume_parser.process_file(resume_path, user_id, save_json=True)
            except Exception as e:
                print(f"Error processing {resume_file}: {str(e)}")