from matcher_model.matcher import Matcher
from matcher_model.utils.resume_loader import load_resume_statements
from matcher_model.models.data_models import JobRequirement
from matcher_model.utils.logger import setup_logger
from matcher_model.utils.result_saver import save_matching_results
import json
from pathlib import Path

if __name__ == "__main__":
    # Setup logging
    logger = setup_logger("matcher", level="INFO", log_file="matcher.log")
    
    # Initialize matcher
    logger.info("Initializing matcher")
    matcher = Matcher()
    
    # Load resumes from the directory structure
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
    
    logger.info(f"Loading resumes from {base_path}")
    resumes = load_resume_statements(base_path, categories)
    logger.info(f"Loaded {len(resumes)} resumes")
    
    # Initialize vector store with all resume statements
    logger.info("Initializing vector store")
    all_statements = []
    for resume in resumes:
        statements = matcher.statement_processor.prepare_statements(resume['statements'])
        all_statements.extend(statements)
    
    logger.info(f"Prepared {len(all_statements)} statements for vector store")
    matcher.statement_processor.initialize_vector_store(all_statements)
    
    # Load the real job description from JSON file
    job_json_path = "matcher_dataset/job_descriptions/statements/format_json/63ca5bad3a2fb6111a880f6a.json"
    logger.info(f"Loading job description from {job_json_path}")
    
    with open(job_json_path, 'r', encoding='utf-8') as f:
        job_data = json.load(f)
    
    # Convert JSON data to JobRequirement objects with appropriate categories
    must_have_requirements = []
    
    # Add core must-have requirements
    must_have_requirements.extend([
        JobRequirement(text=req, type="must_have", category="requirement")
        for req in job_data['must_have_requirements']
    ])
    
    # Add required skills as must-have
    must_have_requirements.extend([
        JobRequirement(text=f"Proficiency in {skill}", type="must_have", category="skill")
        for skill in job_data['required_skills']
    ])
    
    # Add experience requirements as must-have
    must_have_requirements.extend([
        JobRequirement(text=exp, type="must_have", category="experience")
        for exp in job_data['experience_required']
    ])
    
    # Add educational requirements as must-have
    must_have_requirements.extend([
        JobRequirement(text=edu, type="must_have", category="education")
        for edu in job_data['educational_requirements']
    ])
    
    nice_to_have_requirements = []
    
    # Add core nice-to-have requirements
    nice_to_have_requirements.extend([
        JobRequirement(text=req, type="nice_to_have", category="requirement")
        for req in job_data['nice_to_have_requirements']
    ])
    
    # Add responsibilities as nice-to-have (matching experience with these responsibilities)
    nice_to_have_requirements.extend([
        JobRequirement(text=f"Experience with: {resp}", type="nice_to_have", category="responsibility")
        for resp in job_data['responsibilities']
    ])
    
    job_description = {
        'must_have_requirements': must_have_requirements,
        'nice_to_have_requirements': nice_to_have_requirements
    }
    
    # Run matching
    logger.info("Starting matching process")
    matches = matcher.match(job_description, resumes)
    logger.info(f"Found {len(matches)} matching resumes")
    
    # Save results
    json_path, txt_path = save_matching_results(matches, job_description)
    
    # Print summary to console
    logger.info("\nTop 5 Matches:")
    for match in matches[:5]:
        logger.info(f"Resume {match['id']}: {match['score']:.2%} match ({match['category']})")