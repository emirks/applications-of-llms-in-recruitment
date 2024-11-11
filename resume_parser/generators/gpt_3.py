from . import BaseGenerator

from llama_index.llms.openai import OpenAI

client = OpenAI(model="gpt-3.5-turbo-0125", max_tokens=2000)

class GPT3Generator(BaseGenerator):
    def generate_json(self, text):
        return super().generate_json(text, client)
    
    def improve_by_reiteration(self, text):
        return super().improve_by_reiteration(text, client)