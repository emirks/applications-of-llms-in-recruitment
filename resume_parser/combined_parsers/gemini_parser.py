from . import BaseParser, BaseGenerator, json_format

import json
import json_repair
import google.generativeai as genai
import os

genai.configure(api_key=os.environ['GEMINI_API_KEY'])
model = genai.GenerativeModel(model_name="gemini-1.5-flash", 
                              generation_config={"response_mime_type": "application/json"})

prompt = f"""
Get the CV data from the given CV images or CV pdf in the following json format.
Think images as a part of the same CV document if there are multiple images.

Json format:
{json_format}
"""

class GeminiParser(BaseParser, BaseGenerator):
    def parse_text(self, path, images):
        file = genai.upload_file(path=path, display_name=path)
        response = model.generate_content([file, prompt])

        cv_content = response.text
        try: 
            cv_content_json = json.loads(cv_content)
        except json.decoder.JSONDecodeError:
            cv_content_json = json_repair.loads(cv_content)
        
        print(response.usage_metadata)

        genai.delete_file(file.name)
        return cv_content_json

    def generate_json(self, parsed_text):
        return parsed_text