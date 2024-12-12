import yaml
import importlib
import json
import os
import logging
from typing import Dict, Any, List
from .generators import BaseGenerator
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_PIPELINE_NAME = "geminiparser_gpt4o"

class StatementExtractor:
    def __init__(self, pipeline_name=DEFAULT_PIPELINE_NAME):
        self.pipeline_name = pipeline_name
        logger.info(f"Initializing StatementExtractor with pipeline: {self.pipeline_name}")
        self.config = self.read_pipeline_yaml()
        self.generator = self.create_generator_from_pipeline_config(self.config)
        
    def read_pipeline_yaml(self):
        yaml_path = f"resume_statement_extractor/yaml_configs/{self.pipeline_name}.yaml"
        try:
            with open(yaml_path, 'r') as file:
                config = yaml.safe_load(file)
                logger.info(f"Successfully loaded config from {yaml_path}")
                return config
        except Exception as e:
            logger.error(f"Failed to read pipeline YAML {yaml_path}: {e}")
            raise

    def create_generator_from_pipeline_config(self, config):
        try:
            generator_path = config['generator'].split('.')[0]
            generator_module = importlib.import_module(f"resume_statement_extractor.generators.{generator_path}")
            GeneratorClass = getattr(generator_module, config['generator'].split('.')[1])
            logger.info(f"Creating generator: {config['generator']}")
            return GeneratorClass()
        except Exception as e:
            logger.error(f"Failed to create generator {config['generator']}: {e}")
            raise

    def extract_statements(self, txt_file_path: str) -> Dict[str, Any]:
        try:
            logger.info(f"Starting statement extraction for: {txt_file_path}")
            
            # Read the text file
            with open(txt_file_path, 'r', encoding='utf-8') as f:
                resume_text = f.read()
            logger.info(f"Read {len(resume_text)} characters from file")
            
            # 1. Extract basic information
            logger.info("Extracting basic information")
            basic_info = self.generator.generate_json(self.config['basic_info_prompt'], resume_text)
            
            # 2. Extract skills with experience
            logger.info("Extracting skills with experience")
            skills = self.generator.generate_json(self.config['skills_with_experience_prompt'], resume_text)
            
            # 3. Combine results
            result = {
                "personal_info": basic_info.get("personal_info", []),
                "education": basic_info.get("education", []),
                "certifications": basic_info.get("certifications", []),
                "personality_traits": basic_info.get("personality_traits", []),
                "skills": skills.get("skills", [])
            }
            
            logger.info(f"Completed statement extraction with {len(result['skills'])} skills")
            return result
            
        except Exception as e:
            logger.error(f"Failed to extract statements from {txt_file_path}: {e}")
            raise

    def save_statements(self, statements: Dict[str, Any], category: str, filename: str):
        logger.info(f"Saving statements for {category}/{filename}")
        
        # Create output directories if they don't exist
        json_dir = os.path.join("matcher_dataset", "resume", "statements", "format_json", category)
        txt_dir = os.path.join("matcher_dataset", "resume", "statements", "format_txt", category)
        os.makedirs(json_dir, exist_ok=True)
        os.makedirs(txt_dir, exist_ok=True)

        try:
            # Save JSON
            json_path = os.path.join(json_dir, f"{filename}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(statements, f, indent=4)
            logger.info(f"Saved JSON to {json_path}")

            # Save TXT (formatted version)
            txt_path = os.path.join(txt_dir, f"{filename}.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                # Create progress bar for formatting
                skills = statements.get("skills", [])
                formatted_sections = []
                
                # Format basic sections
                formatted_sections.extend([
                    "Personal Information:",
                    *[f"- {s}" for s in statements.get("personal_info", [])],
                    "\nEducation:",
                    *[f"- {s}" for s in statements.get("education", [])],
                    "\nCertifications:",
                    *[f"- {s}" for s in statements.get("certifications", [])],
                    "\nPersonality Traits:",
                    *[f"- {s}" for s in statements.get("personality_traits", [])]
                ])
                
                # Format skills with progress tracking
                if skills:
                    formatted_sections.append("\nSkills and Experience:")
                    for skill in tqdm(skills, desc="Formatting skills", leave=False):
                        formatted_sections.extend([
                            f"\n{skill['name']}:",
                            f"- Years of Experience: {skill['years']}",
                            f"- Proficiency Level: {skill['level']}",
                            f"- Summary: {skill['description']}",
                            "- Supporting Evidence:",
                            *[f"  * {e}" for e in skill['evidence']]
                        ])
                
                f.write("\n".join(formatted_sections))
            logger.info(f"Saved formatted text to {txt_path}")
            
        except Exception as e:
            logger.error(f"Error saving statements for {category}/{filename}: {e}")
            raise

    def format_statements(self, statements: Dict[str, Any]) -> str:
        output = []
        
        # Personal Information
        output.append("Personal Information:")
        for statement in statements["personal_info"]:
            output.append(f"- {statement}")
        
        # Education
        output.append("\nEducation:")
        for edu in statements["education"]:
            output.append(f"- {edu}")
        
        # Certifications
        output.append("\nCertifications:")
        for cert in statements["certifications"]:
            output.append(f"- {cert}")
        
        # Personality Traits
        output.append("\nPersonality Traits:")
        for trait in statements["personality_traits"]:
            output.append(f"- {trait}")
        
        # Skills (with evidence)
        output.append("\nSkills and Experience:")
        for skill in statements["skills"]:
            output.append(f"\n{skill['name']}:")
            output.append(f"- Years of Experience: {skill['years']}")
            output.append(f"- Proficiency Level: {skill['level']}")
            output.append(f"- Summary: {skill['description']}")
            output.append("- Supporting Evidence:")
            for evidence in skill['evidence']:
                output.append(f"  * {evidence}")
        
        return "\n".join(output)
