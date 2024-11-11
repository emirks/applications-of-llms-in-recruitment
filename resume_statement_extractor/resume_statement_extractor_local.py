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

DEFAULT_PIPELINE_NAME = "geminiparser_gpt4o"

class StatementExtractor:
    def __init__(self, pipeline_name=DEFAULT_PIPELINE_NAME):
        self.pipeline_name = pipeline_name
        logger.info(f"Initializing StatementExtractor with pipeline: {self.pipeline_name}")
        self.config = self.read_pipeline_yaml()
        self.reader, self.parser, self.generator = self.create_models_from_pipeline_config(self.config)

    def read_pipeline_yaml(self):
        yaml_path = f"resume_statement_extractor/yaml_configs/{self.pipeline_name}.yaml"
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
            reader_module = importlib.import_module(f"resume_statement_extractor.readers.{config['reader'].split('.')[0]}")
            ReaderClass = getattr(reader_module, config['reader'].split('.')[1])

            is_combined_parser = 'combined_parser' in config
            if is_combined_parser:
                parser_module = importlib.import_module(f"resume_statement_extractor.combined_parsers.{config['combined_parser'].split('.')[0]}")
                generator_module = importlib.import_module(f"resume_statement_extractor.combined_parsers.{config['combined_parser'].split('.')[0]}")
                ParserClass = getattr(parser_module, config['combined_parser'].split('.')[1])
                GeneratorClass = getattr(generator_module, config['combined_parser'].split('.')[1])
            else: 
                parser_module = importlib.import_module(f"resume_statement_extractor.parsers.{config['parser'].split('.')[0]}")
                generator_path = config['generator'].split('.')[0]
                generator_folders = ["resume_statement_extractor.generators"]
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

    def extract_statements(self, file_path: str, parsed_json: Dict[str, Any] = None) -> Dict[str, Any]:
        if parsed_json:
            logger.info("Processing from parsed JSON")
            statements = self.generator.generate_json_w_parsed_json(
                resume_text="",  # Empty since we're using parsed JSON
                parsed_json_output=json.dumps(parsed_json),
                client=self.generator.client
            )
            logger.info("Generated statements from parsed JSON")
        else:
            logger.info(f"Processing file: {file_path}")
            images = self.reader.read_image(file_path)
            if images is None:
                logger.error(f"Failed to load image from {file_path}")
                return None

            parsed_text = self.parser.parse_text(file_path, images)
            logger.info(f"Parsed text successfully from {file_path}")

            statements = self.generator.generate_json(parsed_text)
            logger.info(f"Generated statements from text")
        
        return StatementData(**statements).dict()

    def format_output(self, statements: Dict[str, Any]) -> str:
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract statements from resume')
    parser.add_argument('file_path', type=str, help='Path to the resume file')
    parser.add_argument('--pipeline', type=str, default=DEFAULT_PIPELINE_NAME, help='Name of the pipeline to use')
    args = parser.parse_args()

    extractor = StatementExtractor(args.pipeline)
    statements = extractor.extract_statements(args.file_path)
    formatted_output = extractor.format_output(statements)
    print(formatted_output)
