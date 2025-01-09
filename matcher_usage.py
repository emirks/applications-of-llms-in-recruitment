from matcher_model.matcher import Matcher
from matcher_model.utils.resume_loader import load_resume_statements
from matcher_model.models.data_models import JobRequirement
from matcher_model.utils.logger import setup_logger
from matcher_model.utils.result_saver import save_matching_results
import json
from pathlib import Path

# Load the real job description from JSON file
JOB_DESCRIPTION_PATH = "matcher_dataset/job_descriptions/statements/format_json/63ca7a693a2fb6111a882eb4.json"


if __name__ == "__main__":
    # Setup logging
    logger = setup_logger("matcher_model", level="INFO", log_file="matcher.log")
    
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
        statements = matcher.statement_processor.prepare_statements(
            resume['statements'],
            resume_id=resume['id']
        )
        all_statements.extend(statements)
    
    logger.info(f"Prepared {len(all_statements)} statements for vector store")
    # After preparing statements
    logger.info(f"Sample of prepared statements:")
    for stmt in all_statements[:5]:
        logger.info(f"Resume ID: {stmt['resume_id']}, Text: {stmt['text'][:100]}")

    # Force recreation of vector store
    matcher.statement_processor.initialize_vector_store(all_statements, force_recreate=False)
    
        logger.info(f"Loading job description from {JOB_DESCRIPTION_PATH}")
    
    with open(JOB_DESCRIPTION_PATH, 'r', encoding='utf-8') as f:
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
    matches = matcher.match_new(job_description, resumes)
    logger.info(f"Found {len(matches)} matching resumes")
    
    # Save results
    json_path, txt_path = save_matching_results(matches, job_description)
    
    # Print summary to console
    logger.info("\nTop 5 Matches:")
    for match in matches[:5]:
        logger.info(f"Resume matcher_dataset/resume/statements/format_json/{match['id']}: {match['score']:.2%} match ({match['category']})")
    
    