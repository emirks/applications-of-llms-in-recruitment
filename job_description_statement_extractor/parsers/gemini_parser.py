from . import BaseParser

import os
import PIL.Image
import google.generativeai as genai

genai.configure(api_key=os.environ["DEV_GEMINI_API_KEY"])
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
)

prompt = """
Extract the text from the CV document. Get all the text available in the document.
Return only the extracted text. 
"""

class GeminiParser(BaseParser):
    def parse_text(self, pdf_file, images):
        if isinstance(pdf_file, str):
            file = genai.upload_file(path=pdf_file, display_name=pdf_file)
        else:
            file = PIL.Image.open(pdf_file)

        response = model.generate_content([file, prompt])

        cv_content = response.text
        print(response.usage_metadata)

        genai.delete_file(file.name)
        return cv_content
