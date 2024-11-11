from . import BaseGenerator

import os
from llama_index.llms.groq import Groq

client = Groq(model="llama-3.1-70b-versatile", api_key=os.environ.get("GROQ_API_KEY"))

class Llama70bGenerator(BaseGenerator):
    def generate_json(self, text):
        return super().generate_json(text, client)
    
    def improve_by_reiteration(self, text):
        return super().improve_by_reiteration(text, client)
