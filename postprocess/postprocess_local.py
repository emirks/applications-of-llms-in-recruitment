"""
This module provides a class `ResumeJsonPostProcess` for processing and enhancing JSON files representing resumes.

The `ResumeJsonPostProcess` class reads a JSON file and its corresponding raw text file, processes specific fields using
the `ExperienceProcessor` and `GeneralProcessor` components, and then updates the JSON file with the processed data.

Attributes:
    PATH_JSON_DATABASE (str): The path to the directory containing the original JSON files.
    PATH_PROCESSED_JSON_DATABASE (str): The path to the directory where the processed JSON files will be saved.
    PATH_TXT_DATABASE (str): The path to the directory containing the raw text files.

Classes:
    ResumeJsonPostProcess: Handles the reading, processing, and writing of resume JSON files.
"""

from ai_systems.postprocess.components.postprocess_experience import ExperienceProcessor
from ai_systems.postprocess.components.postprocess_general import GeneralResumeProcessor, GeneralApplicationProcessor
from typing_extensions import List, Dict
import logging
import os
import json
from dotenv import load_dotenv
load_dotenv(override=True)

path_resume_database = os.getenv('DO_SPACE_PATH_RESUME_DATABASE')
path_application_database = os.getenv('DO_SPACE_PATH_APPLICATION_DATABASE')

class ResumeJsonPostProcess:
    def __init__(self) -> None:
        self.experience_processor = ExperienceProcessor()
        self.general_processor = GeneralResumeProcessor()
        self.processed_fields = {}
    
    def read_data(self, file_name_no_suffix, user_id):
        self.user_id = user_id
        self.file_name = file_name_no_suffix

        self.base_json = self.read_json()
        self.processed_json = self.base_json
        self.raw_text = self.read_txt()

    def process(self, file_name_no_suffix, user_id):
        self.read_data(file_name_no_suffix, user_id)

        experience_fields = self.experience_processor.process(self.processed_json)
        general_fields = self.general_processor.process(self.raw_text, self.user_id)

        self.processed_fields ={
            'experience_fields': experience_fields,
            'general_fields': general_fields,
        }
        self.add_processed_fields_to_json()
        self.write_json()
        return self.processed_json

    def add_processed_fields_to_json(self):
        for processed_field_name, processed_field_dict in self.processed_fields.items():
            for key, value in processed_field_dict.items():
                self.processed_json[key] = value
                logging.debug(f"Field added to JSON: {key}: {value}")

    def read_json(self):
        # Read JSON
        with open(os.path.join(path_resume_database, 'format_json', self.file_name + '.json'), 'r') as file:
            return json.load(file)
        
    def write_json(self):
        # Write JSON
        with open(os.path.join(path_resume_database, 'format_json_processed', self.file_name + '.json'), 'w') as file:
            json.dump(self.processed_json, file, indent=4)     

    def read_txt(self):
        # Read TXT
        with open(os.path.join(path_resume_database, 'format_txt', self.file_name + '.txt'), 'r') as file:
            return file.read()


class ApplicationJsonPostProcess:
    def __init__(self) -> None:
        self.experience_processor = ExperienceProcessor()
        self.general_processor = GeneralApplicationProcessor()
        self.processed_fields = {}
    
    def read_data(self, file_name_no_suffix, job_id):
        self.job_id = job_id
        self.file_name = file_name_no_suffix

        self.base_json = self.read_json()
        self.processed_json = self.base_json
        self.raw_text = self.read_txt()

    def process(self, file_name_no_suffix, job_id):
        self.read_data(file_name_no_suffix, job_id)
        general_fields = self.general_processor.process(self.raw_text, self.job_id)

        self.processed_fields ={
            'general_fields': general_fields,
        }
        self.add_processed_fields_to_json()
        self.write_json()
        return self.processed_json

    def add_processed_fields_to_json(self):
        for processed_field_name, processed_field_dict in self.processed_fields.items():
            for key, value in processed_field_dict.items():
                self.processed_json[key] = value
                logging.debug(f"Field added to JSON: {key}: {value}")

    def read_json(self):
        # Read JSON
        with open(os.path.join(path_application_database, 'format_json', self.file_name + '.json'), 'r') as file:
            return json.load(file)
        
    def write_json(self):
        # Write JSON
        with open(os.path.join(path_application_database, 'format_json_processed', self.file_name + '.json'), 'w') as file:
            json.dump(self.processed_json, file, indent=4)     

    def read_txt(self):
        # Read TXT
        with open(os.path.join(path_application_database, 'format_txt', self.file_name + '.txt'), 'r') as file:
            return file.read()