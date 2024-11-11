from . import BaseParser, BaseGenerator, json_format

import io
import base64
import json
import json_repair
from openai import OpenAI

client = OpenAI()

prompt = f"""
Get the CV data from the given CV images in the following json format.
Think images as a part of the same CV document if there are multiple images.
{json_format}
"""

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt,
            },
        ],
    }
]

class Gpt4oParser(BaseParser, BaseGenerator):
    def parse_text(self, pdf_path, images):
        base64_images = []

        for image in images:
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

            messages[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                }
            )
            base64_images.append(base64_image)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=2000,
            response_format={"type": "json_object"},
        )

        cv_content = response.choices[0].message.content
        try: 
            cv_content_json = json.loads(cv_content)
        except json.decoder.JSONDecodeError:
            cv_content_json = json_repair.loads(cv_content)

        return cv_content_json

    def generate_json(self, parsed_text):
        return parsed_text