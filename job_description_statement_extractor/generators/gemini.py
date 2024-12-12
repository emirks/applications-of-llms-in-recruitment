from . import BaseGenerator, prompt, json_format
import google.generativeai as genai
import os
import json
import re

class GeminiGenerator(BaseGenerator):
    def __init__(self):
        genai.configure(api_key=os.environ.get("DEV_GEMINI_API_KEY"))
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        self.generation_config = {
            "temperature": 0.1,
            "max_output_tokens": 4000,
        }

    def generate_json(self, text):
        formatted_prompt = prompt.format(json_format=json_format, cv_text=text)
        response = self.model.generate_content(
            formatted_prompt,
            generation_config=self.generation_config
        )
        
        # Extract JSON from response
        json_match = re.search(r'{[\s\S]*}', response.text)
        if json_match:
            cv_json_str = json_match.group(0)
            return json.loads(cv_json_str)
        else:
            print("No JSON found in the text.")
            return None
    
    def generate_json_w_parsed_json(self, resume_text, parsed_json_output, client=None):
        formatted_prompt = prompt_w_parsed_json.format(
            json_format_schema=json_format,
            resume_text=resume_text,
            parsed_json_output=parsed_json_output
        )
        response = self.model.generate_content(
            formatted_prompt,
            generation_config=self.generation_config
        )
        
        # Extract JSON from response
        json_match = re.search(r'{[\s\S]*}', response.text)
        if json_match:
            cv_json_str = json_match.group(0)
            return json.loads(cv_json_str)
        else:
            print("No JSON found in the text.")
            return None
    
    def improve_by_reiteration(self, text):
        return super().improve_by_reiteration(text, self.model)
