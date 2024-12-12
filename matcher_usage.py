from matcher_model.matcher import Matcher
from matcher_model.utils.resume_loader import load_resume_statements
from matcher_model.models.data_models import JobRequirement
from matcher_model.utils.logger import setup_logger
from matcher_model.utils.result_saver import save_matching_results

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
    
    # Example job description
    logger.info("Setting up job requirements")
    job_description = {
        'must_have_requirements': [
            JobRequirement(text="Python programming experience", type="must_have", category="skill"),
            JobRequirement(text="Experience with web development", type="must_have", category="skill")
        ],
        'nice_to_have_requirements': [
            JobRequirement(text="Experience with React", type="nice_to_have", category="skill"),
            JobRequirement(text="Knowledge of MongoDB", type="nice_to_have", category="skill")
        ]
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