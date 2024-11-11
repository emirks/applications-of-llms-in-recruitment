from . import BaseGenerator
from llama_index.llms.openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(override=True)
api_key = os.getenv('DEV_OPENAI_API_KEY')

os.environ['OPENAI_API_KEY'] = api_key

client = OpenAI(model="gpt-4", api_key=os.environ.get("OPENAI_API_KEY"))

class GPT4oGenerator(BaseGenerator):
    def __init__(self):
        super().__init__()
        self.client = client
    
    def generate_json(self, text):
        return super().generate_json(text, self.client)
    
    def improve_by_reiteration(self, text):
        return super().improve_by_reiteration(text, self.client)
    
    def generate_json_w_parsed_json(self, resume_text, parsed_json_output, client):
        return super().generate_json_w_parsed_json(resume_text, parsed_json_output, self.client)