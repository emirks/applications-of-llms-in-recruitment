import yaml
import importlib
import json
import argparse
import os
import logging
from typing import Dict, Any

from .readers import BaseReader
from .generators import BaseGenerator
from .parsers import BaseParser
from .response_model import StatementData

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_PIPELINE_NAME = "textparser_gpt4o"

class StatementExtractor:
    def __init__(self, pipeline_name=DEFAULT_PIPELINE_NAME):
        self.pipeline_name = pipeline_name
        logger.info(f"Initializing StatementExtractor with pipeline: {self.pipeline_name}")
        self.config = self.read_pipeline_yaml()
        self.reader, self.parser, self.generator = self.create_models_from_pipeline_config(self.config)

    def read_pipeline_yaml(self):
        yaml_path = f"job_description_statement_extractor/yaml_configs/{self.pipeline_name}.yaml"
        logger.info(f"Reading pipeline configuration from {yaml_path}")
        try:
            with open(yaml_path, 'r') as file:
                config = yaml.safe_load(file)
                logger.info(f"Pipeline configuration loaded successfully")
                return config
        except Exception as e:
            logger.error(f"Failed to read pipeline YAML: {e}")
            raise

    def create_models_from_pipeline_config(self, config):
        logger.info("Creating models from pipeline configuration")
        try:
            reader_module = importlib.import_module(f"job_description_statement_extractor.readers.{config['reader'].split('.')[0]}")
            ReaderClass = getattr(reader_module, config['reader'].split('.')[1])

            parser_module = importlib.import_module(f"job_description_statement_extractor.parsers.{config['parser'].split('.')[0]}")
            generator_path = config['generator'].split('.')[0]
            generator_folders = ["job_description_statement_extractor.generators"]
            
            for folder in generator_folders:
                try:
                    generator_module = importlib.import_module(f"{folder}.{generator_path}")
                    break
                except ModuleNotFoundError:
                    raise
                
            ParserClass = getattr(parser_module, config['parser'].split('.')[1])
            GeneratorClass = getattr(generator_module, config['generator'].split('.')[1])

            reader : BaseReader = ReaderClass()
            parser : BaseParser = ParserClass()
            generator : BaseGenerator = GeneratorClass()

            logger.info("Models created successfully")
            return reader, parser, generator
        except Exception as e:
            logger.error(f"Failed to create models from pipeline configuration: {e}")
            raise

    def extract_statements(self, file_path: str) -> Dict[str, Any]:
        logger.info(f"Processing file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                job_description_text = f.read()
        except Exception as e:
            logger.error(f"Failed to read text file {file_path}: {str(e)}")
            return None


        statements = self.generator.generate_json(job_description_text)
        logger.info(f"Generated statements from text")
        
        return StatementData(**statements).dict()

    def format_output(self, statements: Dict[str, Any]) -> str:
        output = []
        
        # Job Information
        for info in statements["job_info"]:
            output.append(info)
        
        # Must-Have Requirements
        output.append("\nMust-Have Requirements:")
        for req in statements["must_have_requirements"]:
            output.append(f"- {req}")
        
        # Nice-to-Have Requirements
        output.append("\nNice-to-Have Requirements:")
        for req in statements["nice_to_have_requirements"]:
            output.append(f"- {req}")
        
        # Responsibilities
        output.append("\nResponsibilities:")
        for resp in statements["responsibilities"]:
            output.append(f"- {resp}")
        
        # Required Skills
        output.append("\nRequired Skills:")
        for skill in statements["required_skills"]:
            output.append(f"- {skill}")
        
        # Experience Required
        output.append("\nExperience Required:")
        for exp in statements["experience_required"]:
            output.append(f"- {exp}")
        
        # Educational Requirements
        output.append("\nEducational Requirements:")
        for edu in statements["educational_requirements"]:
            output.append(f"- {edu}")
        
        # Additional Information
        output.append("\nAdditional Information:")
        for info in statements["additional_info"]:
            output.append(f"- {info}")
        
        return "\n".join(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract statements from resume')
    parser.add_argument('file_path', type=str, help='Path to the resume file')
    parser.add_argument('--pipeline', type=str, default=DEFAULT_PIPELINE_NAME, help='Name of the pipeline to use')
    args = parser.parse_args()

    extractor = StatementExtractor(args.pipeline)
    statements = extractor.extract_statements(args.file_path)
    formatted_output = extractor.format_output(statements)
    print(formatted_output)
