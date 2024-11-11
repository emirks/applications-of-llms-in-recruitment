from . import BaseParser, BaseGenerator, json_format, empty_json

import os
import asyncio
import nest_asyncio

from llama_parse import LlamaParse
import json
import json_repair

nest_asyncio.apply()

prompt = f"""
Get the CV data from the document in the following format.
If the data is not available, leave the field empty.
{json_format}
"""

parser = LlamaParse(
    result_type="markdown",
    language="en",
    parsing_instruction=prompt,
    invalidate_cache=True,
)

class LlamaParser(BaseParser, BaseGenerator):
    async def parse_and_generate_async(self, pdf_path, images):
        response_json = json.loads(empty_json)
        for ind, image in enumerate(images):
            image_name = pdf_path.split("/")[-1].split(".")[0]
            tmp_img_path = f"../images/{image_name}_{ind}.jpg"
            image.save(tmp_img_path, format="JPEG")
            documents = await parser.aload_data(tmp_img_path)
            if not documents or not len(documents):
                print("No CV content parsed")
                return
            image_response = json_repair.loads(documents[0].text)
            if not image_response:
                continue
            for key, value in image_response.items():
                if value == "" or value == [] or value == {}:
                    continue
                # check if value is a string and it is not empty
                if isinstance(value, str) and response_json[key] != "":
                    continue
                if isinstance(response_json[key], list):
                    response_json[key].extend(value)
                else:
                    response_json[key] = value
            
            os.remove(tmp_img_path)
        return response_json
    
    def parse_text(self, pdf_path, images):
        return asyncio.run(self.parse_and_generate_async(pdf_path, images))
    
    def generate_json(self, parsed_text):
        return parsed_text