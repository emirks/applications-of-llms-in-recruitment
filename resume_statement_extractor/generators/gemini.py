from . import BaseGenerator

from llama_index.llms.gemini import Gemini
import os

client = Gemini(
    model="models/gemini-1.5-pro",
    api_key=os.environ.get("DEV_GEMINI_API_KEY"),
    generation_config={"response_mime_type": "application/json"},
)


class GeminiGenerator(BaseGenerator):
    def generate_json(self, text):
        return super().generate_json(text, client)
    
    def improve_by_reiteration(self, text):
        return super().improve_by_reiteration(text, client)
