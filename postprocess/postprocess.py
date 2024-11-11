from ai_systems.postprocess.components.postprocess_experience import ExperienceProcessor
from ai_systems.postprocess.components.postprocess_general import GeneralResumeProcessor, GeneralApplicationProcessor
from typing_extensions import List, Dict
import logging
import os
import json
import boto3
from dotenv import load_dotenv
load_dotenv(override=True)

# Define paths to the directories in your DigitalOcean Space
path_resume_database = os.getenv('DO_SPACE_PATH_RESUME_DATABASE')
path_application_database = os.getenv('DO_SPACE_PATH_APPLICATION_DATABASE')

class ResumeJsonPostProcess:
    def __init__(self) -> None:
        self.experience_processor = ExperienceProcessor()
        self.general_processor = GeneralResumeProcessor()
        self.processed_fields = {}
        logging.info("Initialized ResumeJsonPostProcess.")

    def process(self, json_output, raw_text, user_id):
        logging.info(f"Starting postprocess for user ID: {user_id}")

        # Process the experience and general fields
        experience_fields = self.experience_processor.process(json_output)
        general_fields = self.general_processor.process(raw_text, user_id)

        # Combine processed fields with the original JSON output
        self.processed_fields = {
            'experience_fields': experience_fields,
            'general_fields': general_fields,
        }
        self.add_processed_fields_to_json(json_output)
        
        return json_output  # Return the updated JSON

    def add_processed_fields_to_json(self, json_output):
        """Add processed fields to the original JSON."""
        for processed_field_name, processed_field_dict in self.processed_fields.items():
            for key, value in processed_field_dict.items():
                json_output[key] = value
                logging.debug(f"Field added to JSON: {key}: {value}")



class ApplicationJsonPostProcess:
    def __init__(self) -> None:
        self.general_processor = GeneralApplicationProcessor()
        self.processed_fields = {}
        logging.info("Initialized ApplicationJsonPostProcess.")

    def process(self, json_output, raw_text, job_id):
        logging.info(f"Starting postprocess for job ID: {job_id}")

        # Process the general fields
        general_fields = self.general_processor.process(raw_text, job_id)

        # Combine processed fields with the original JSON output
        self.processed_fields = {
            'general_fields': general_fields,
        }
        self.add_processed_fields_to_json(json_output)
        
        return json_output  # Return the updated JSON

    def add_processed_fields_to_json(self, json_output):
        """Add processed fields to the original JSON."""
        for processed_field_name, processed_field_dict in self.processed_fields.items():
            for key, value in processed_field_dict.items():
                json_output[key] = value
                logging.debug(f"Field added to JSON: {key}: {value}")
