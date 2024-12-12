import yaml
import importlib
import json
import os
import logging
from typing import Dict, Any
from .generators import BaseGenerator

logging.basicConfig(level=logging.INFO)
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
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Failed to read pipeline YAML: {e}")
            raise

    def create_generator_from_pipeline_config(self, config):
        try:
            generator_path = config['generator'].split('.')[0]
            generator_module = importlib.import_module(f"resume_statement_extractor.generators.{generator_path}")
            GeneratorClass = getattr(generator_module, config['generator'].split('.')[1])
            return GeneratorClass()
        except Exception as e:
            logger.error(f"Failed to create generator: {e}")
            raise

    def extract_statements(self, txt_file_path: str) -> Dict[str, Any]:
        try:
            # Read the text file
            with open(txt_file_path, 'r', encoding='utf-8') as f:
                resume_text = f.read()
            
            # Generate statements
            statements = self.generator.generate_json(resume_text)
            logger.info(f"Generated statements from {txt_file_path}")
            
            return statements
        except Exception as e:
            logger.error(f"Failed to extract statements from {txt_file_path}: {e}")
            raise

    def save_statements(self, statements: Dict[str, Any], category: str, filename: str):
        # Create output directories if they don't exist
        json_dir = os.path.join("matcher_dataset", "resume", "statements", "format_json", category)
        txt_dir = os.path.join("matcher_dataset", "resume", "statements", "format_txt", category)
        os.makedirs(json_dir, exist_ok=True)
        os.makedirs(txt_dir, exist_ok=True)

        # Save JSON
        json_path = os.path.join(json_dir, f"{filename}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(statements, f, indent=4)

        # Save TXT
        txt_path = os.path.join(txt_dir, f"{filename}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(self.format_statements(statements))

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
        
        # Skills
        output.append("\nSkills:")
        for skill in statements["skills"]:
            output.append(f"- {skill}")
        
        return "\n".join(output)
