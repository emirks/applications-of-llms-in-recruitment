from . import BaseGenerator

import os
from llama_index.llms.groq import Groq

client = Groq(model="gemma2-9b-it", api_key=os.environ.get("GROQ_API_KEY"))

class Gemma9bGenerator(BaseGenerator):
    def generate_json(self, text):
        return super().generate_json(text, client)
    
    def improve_by_reiteration(self, text):
        return super().improve_by_reiteration(text, client)