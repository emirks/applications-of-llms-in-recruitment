import yaml
import importlib
import json
import argparse
import os
import logging

from resume_parser.readers import BaseReader
from resume_parser.generators import BaseGenerator
from resume_parser.parsers import BaseParser

from ai_systems.postprocess.postprocess import ResumeJsonPostProcess

from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_PIPELINE_NAME = "geminiparser_gpt4o"

class ResumeParser: 
    def __init__(self, pipeline_name=DEFAULT_PIPELINE_NAME):
        self.pipeline_name = pipeline_name
        logger.info(f"Initializing ResumeParser with pipeline: {self.pipeline_name}")
        self.config = self.read_pipeline_yaml()
        self.reader, self.parser, self.generator = self.create_models_from_pipeline_config(self.config)
        self.postprocessor = ResumeJsonPostProcess()

    def read_pipeline_yaml(self):
        yaml_path = f"resume_parser/yaml_configs/{self.pipeline_name}.yaml"
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
            reader_module = importlib.import_module(f"resume_parser.readers.{config['reader'].split('.')[0]}")
            ReaderClass = getattr(reader_module, config['reader'].split('.')[1])

            is_combined_parser = 'combined_parser' in config
            if is_combined_parser:
                parser_module = importlib.import_module(f"resume_parser.combined_parsers.{config['combined_parser'].split('.')[0]}")
                generator_module = importlib.import_module(f"resume_parser.combined_parsers.{config['combined_parser'].split('.')[0]}")
                ParserClass = getattr(parser_module, config['combined_parser'].split('.')[1])
                GeneratorClass = getattr(generator_module, config['combined_parser'].split('.')[1])
            else: 
                parser_module = importlib.import_module(f"resume_parser.parsers.{config['parser'].split('.')[0]}")
                generator_path = config['generator'].split('.')[0]
                generator_folders = ["resume_parser.generators"]
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

    def process_file(self, file_path, user_id, category, save_json=True):
        logger.info(f"Processing file: {file_path} for user: {user_id}")
        images = self.reader.read_image(file_path)
        if images is None:
            logger.error(f"Failed to load image from {file_path}")
            return

        parsed_text = self.parser.parse_text(file_path, images)
        logger.info(f"Parsed text successfully from {file_path}")

        file_name_no_suffix = self._get_file_name_without_suffix(file_path)

        json_output = self.parse_resume(parsed_text)
        logger.info(f"Generated JSON from resume text")

        if save_json:
            self.save_json(file_name_no_suffix, json_output, category)

        return json_output
    
    def save_json(self, json_name: str, response_json, category: str):
        application_data_folder = os.getenv('PATH_RESUME_DATABASE')
        
        # Create the full path including category subfolder
        generated_json_folder = os.path.join(application_data_folder, 'format_json', category)
        
        # Create all necessary directories
        os.makedirs(generated_json_folder, exist_ok=True)
        
        try:      
            # Get just the filename without the category path
            filename = os.path.basename(json_name)
            save_path = os.path.join(generated_json_folder, f"{filename}.json")
            
            with open(save_path, "w") as f:
                json.dump(response_json, f, indent=4)
            logger.info(f"Saved JSON to {save_path}")

        except Exception as e:
            logger.error(f"Failed to save JSON: {e}")
            raise

    def _get_file_name_without_suffix(self, file_path):
        return file_path.split("/")[-1].split(".")[0]

    def extract_text(self, pdf_file):
        logger.info(f"Extracting text from PDF file: {pdf_file}")
        images = self.reader.read_image(pdf_file)
        if images is None:
            logger.error(f"Failed to load image from {pdf_file}")
            return

        parsed_text = self.parser.parse_text(pdf_file, images)
        return parsed_text
    
    def parse_resume(self, resume_text):
        logger.info("Parsing resume text to JSON format")
        return self.generator.generate_json(resume_text)
    

if __name__ == "__main__":
    file_path = "../Fernando_SalinasRomero_1305864826.pdf"
    user_id = "user_1"

    parser = argparse.ArgumentParser(description='Run the pipeline')
    parser.add_argument('pipeline_name', type=str, help='Name of the pipeline to run')
    args = parser.parse_args()

    pipeline_name = args.pipeline_name
    resume_parser = ResumeParser(pipeline_name)
    resume_parser.process_file(file_path, user_id)
