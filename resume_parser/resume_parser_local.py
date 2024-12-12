import yaml
import importlib
import json
import os
import logging
from resume_parser.generators import BaseGenerator
from ai_systems.postprocess.postprocess import ResumeJsonPostProcess
from dotenv import load_dotenv

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_PIPELINE_NAME = "geminiparser_gpt4o"

class ResumeParser: 
    def __init__(self, pipeline_name=DEFAULT_PIPELINE_NAME):
        self.pipeline_name = pipeline_name
        logger.info(f"Initializing ResumeParser with pipeline: {self.pipeline_name}")
        self.config = self.read_pipeline_yaml()
        self.generator = self.create_generator_from_pipeline_config(self.config)
        self.postprocessor = ResumeJsonPostProcess()

    def read_pipeline_yaml(self):
        yaml_path = f"resume_parser/yaml_configs/{self.pipeline_name}.yaml"
        try:
            with open(yaml_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Failed to read pipeline YAML: {e}")
            raise

    def create_generator_from_pipeline_config(self, config):
        try:
            generator_path = config['generator'].split('.')[0]
            generator_module = importlib.import_module(f"resume_parser.generators.{generator_path}")
            GeneratorClass = getattr(generator_module, config['generator'].split('.')[1])
            return GeneratorClass()
        except Exception as e:
            logger.error(f"Failed to create generator: {e}")
            raise

    def process_file(self, file_path, user_id, category, save_json=True):
        """Process a resume from its pre-extracted text file"""
        try:
            # Get text file path
            txt_file_name = os.path.splitext(os.path.basename(file_path))[0] + '.txt'
            txt_file_path = os.path.join(
                'matcher_dataset', 'resume', 'format_txt',
                category, txt_file_name
            )
            
            # Read text and generate JSON
            with open(txt_file_path, 'r', encoding='utf-8') as f:
                resume_text = f.read()
            
            json_output = self.generator.generate_json(resume_text)
            
            if not json_output:
                logger.error(f"Failed to generate JSON for {file_path}")
                return None

            if save_json:
                json_name = os.path.splitext(os.path.basename(file_path))[0]
                output_dir = os.path.join(
                    os.getenv('PATH_RESUME_DATABASE'),
                    'format_json',
                    category
                )
                save_path = os.path.join(output_dir, f"{json_name}.json")
                
                # Verify if JSON was saved
                if os.path.exists(save_path):
                    logger.info(f"JSON already exists: {save_path}")
                else:
                    self.save_json(json_name, json_output, category)
                    logger.info(f"New JSON saved: {save_path}")

            return json_output
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            raise
    
    def save_json(self, json_name: str, response_json, category: str):
        try:
            output_dir = os.path.join(
                os.getenv('PATH_RESUME_DATABASE'),
                'format_json',
                category
            )
            os.makedirs(output_dir, exist_ok=True)
            
            save_path = os.path.join(output_dir, f"{json_name}.json")
            with open(save_path, "w") as f:
                json.dump(response_json, f, indent=4)
                
        except Exception as e:
            logger.error(f"Failed to save JSON: {e}")
            raise

if __name__ == "__main__":
    parser = ResumeParser()
