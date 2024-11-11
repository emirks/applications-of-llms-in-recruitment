import yaml
import importlib
import json
import argparse
import os
import logging
import boto3
import tempfile

from resume_parser.readers import BaseReader
from resume_parser.generators import BaseGenerator
from resume_parser.parsers import BaseParser
from postprocess.postprocess import ResumeJsonPostProcess
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

        # Initialize boto3 client for DigitalOcean Spaces
        self.s3_client = boto3.client(
            's3',
            region_name=os.environ['DO_SPACES_REGION_NAME'],
            aws_access_key_id=os.environ['DO_SPACES_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['DO_SPACES_SECRET_ACCESS_KEY'],
            endpoint_url=os.environ['DO_SPACES_ENDPOINT_URL']
        )
        logger.info("DigitalOcean Spaces client initialized.")

    def read_pipeline_yaml(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(current_dir, "yaml_configs", f"{self.pipeline_name}.yaml")
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

    def fetch_pdf_from_space(self, file_path):
        """Fetches PDF from DigitalOcean Space and saves it locally for processing."""
        resume_data_folder = os.getenv('DO_SPACE_PATH_RESUME_DATABASE')
        s3_key = os.path.join(resume_data_folder, 'format_pdf', file_path).replace("\\", "/")

        try:
            logger.info(f"Fetching PDF from DigitalOcean Space: {s3_key}")
            response = self.s3_client.get_object(Bucket=os.environ['DO_SPACES_BUCKET_NAME'], Key=s3_key)

            # Get a platform-independent temporary directory
            temp_dir = tempfile.gettempdir()
            local_temp_pdf = os.path.join(temp_dir, os.path.basename(file_path))

            with open(local_temp_pdf, 'wb') as f:
                f.write(response['Body'].read())
            logger.info(f"PDF downloaded to temporary file: {local_temp_pdf}")
            return local_temp_pdf
        except Exception as e:
            logger.error(f"Failed to fetch PDF from DigitalOcean Space: {e}")
            return None

    def save_txt_to_space(self, file_name: str, extracted_text: str):
        """Writes the extracted text from the PDF to a .txt file in DigitalOcean Space."""
        resume_data_folder = os.getenv('DO_SPACE_PATH_RESUME_DATABASE')
        target_txt_folder = os.path.join(resume_data_folder, 'format_txt')
        s3_key = os.path.join(target_txt_folder, file_name + '.txt').replace("\\", "/")

        try:
            # Upload the extracted text as a .txt file to DigitalOcean Space
            self.s3_client.put_object(
                Bucket=os.environ['DO_SPACES_BUCKET_NAME'],
                Key=s3_key,
                Body=extracted_text,
                ContentType='text/plain'
            )
            logger.info(f"Successfully uploaded TXT to DigitalOcean Space: {s3_key}")
        except Exception as e:
            logger.error(f"Failed to upload TXT to DigitalOcean Space: {e}")
            raise

    def process_file(self, file_path, user_id, file_name, postprocess=True, save_jsons_to_space=True):
        logger.info(f"Processing file: {file_path} for user: {user_id}")
        local_pdf_path = file_path

        # Proceed with processing the local PDF file
        images = self.reader.read_image(local_pdf_path)
        if images is None:
            logger.error(f"Failed to load image from {local_pdf_path}")

        parsed_text = self.parser.parse_text(local_pdf_path, images)
        logger.info(f"Parsed text successfully from {local_pdf_path}")

        file_name_no_suffix = self._get_file_name_without_suffix(file_name)

        # Save extracted text to DigitalOcean Spaces
        self.save_txt_to_space(file_name_no_suffix, parsed_text)
        logger.info(f"Extracted text from PDF saved to format_txt/{file_name_no_suffix}.txt")

        json_output = self.parse_resume(parsed_text)
        logger.info(f"Generated JSON from resume text")

        if save_jsons_to_space:
            self.save_json_to_space(file_name_no_suffix, json_output)

        if postprocess:
            # Process the JSON using the postprocessor, passing the json_output and raw text
            processed_json = self.postprocessor.process(json_output, parsed_text, user_id)
            
            # Upload the processed JSON to DigitalOcean Spaces
            self.save_processed_json_to_space(file_name_no_suffix, processed_json)
            return processed_json

        return json_output


    def save_json_to_space(self, json_name: str, response_json):
        resume_data_folder = os.getenv('DO_SPACE_PATH_RESUME_DATABASE')
        target_json_folder = os.path.join(resume_data_folder, 'format_json')
        s3_key = os.path.join(target_json_folder, json_name + '.json').replace("\\", "/")

        try:
            json_data = json.dumps(response_json, indent=4)
            self.s3_client.put_object(
                Bucket=os.environ['DO_SPACES_BUCKET_NAME'],
                Key=s3_key,
                Body=json_data,
                ContentType='application/json'
            )
            logger.info(f"Successfully uploaded JSON to DigitalOcean Space: {s3_key}")
        except Exception as e:
            logger.error(f"Failed to upload JSON to DigitalOcean Space: {e}")
            raise

    def save_processed_json_to_space(self, file_name_no_suffix, processed_json):
        """Upload processed JSON to DigitalOcean Space."""
        s3_key = os.path.join(os.getenv('DO_SPACE_PATH_RESUME_DATABASE'), 'format_json_processed', f"{file_name_no_suffix}.json").replace("\\", "/")
        try:
            json_data = json.dumps(processed_json, indent=4)
            self.s3_client.put_object(
                Bucket=os.environ['DO_SPACES_BUCKET_NAME'],
                Key=s3_key,
                Body=json_data,
                ContentType='application/json'
            )
            logger.info(f"Successfully uploaded processed JSON to DigitalOcean Space: {s3_key}")
        except Exception as e:
            logger.error(f"Error uploading processed JSON to DigitalOcean Space: {e}")
            raise

    def _get_file_name_without_suffix(self, file_path):
        return file_path.split("/")[-1].split(".")[0]

    def parse_resume(self, resume_text):
        logger.info("Parsing resume text to JSON format")
        return self.generator.generate_json(resume_text)


if __name__ == "__main__":
    file_path = "emir.pdf"  # The file path in DigitalOcean Space
    user_id = "user_1"

    parser = argparse.ArgumentParser(description='Run the pipeline')
    parser.add_argument('pipeline_name', type=str, help='Name of the pipeline to run')
    args = parser.parse_args()

    pipeline_name = args.pipeline_name
    resume_parser = ResumeParser(pipeline_name)
    resume_parser.process_file(file_path, user_id)
