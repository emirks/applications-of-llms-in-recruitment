from . import BaseGenerator

from llama_index.llms.openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

os.environ['OPENAI_API_KEY'] = api_key
print(api_key)

client = OpenAI(model="gpt-4o-mini", max_tokens=2000)

class GPT4oGenerator(BaseGenerator):
    def generate_json(self, text):
        return super().generate_json(text, client)
    
    def improve_by_reiteration(self, text):
        return super().improve_by_reiteration(text, client)