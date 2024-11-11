from . import BaseParser

import os
import asyncio
import nest_asyncio

from llama_parse import LlamaParse

nest_asyncio.apply()

prompt = """
Extract the text from the CV image. Get all the extracted text. 
"""

parser = LlamaParse(
    result_type="markdown",
    language="en",
    parsing_instruction=prompt,
    invalidate_cache=True,
)


class LlamaParser(BaseParser):
    async def parse_text_async(self, pdf_path, images):
        extracted_text = ""
        if isinstance(pdf_path, str):
            image_prefix = pdf_path.split("/")[-1].split(".")[0]
        else:
            image_prefix = "tmp_resume_image"
        for ind, image in enumerate(images):
            tmp_img_path = f"../images/{image_prefix}_{ind}.jpg"
            image.save(tmp_img_path, format="JPEG")
            documents = await parser.aload_data(tmp_img_path)
            if not documents or not len(documents):
                print("No CV content parsed")
                return
            extracted_text += documents[0].text
            os.remove(tmp_img_path)
        return extracted_text
    
    def parse_text(self, pdf_file, images):
        return asyncio.run(self.parse_text_async(pdf_file, images))