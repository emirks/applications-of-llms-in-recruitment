from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core import Settings
import tiktoken

from llama_index.core.llms import ChatMessage
import json
import json_repair

# types
from llama_index.core.base.llms.types import ChatResponse
from llama_index.core.llms.function_calling import FunctionCallingLLM

import re
from resume_parser.response_model import CVData

class BaseGenerator:
    def __init__(self) -> None:
        super().__init__()
        print("Initializing BaseGenerator: ", self.__class__.__name__)
        print("Embed Model: ", Settings.embed_model.model_name)
        tokenizer = tiktoken.encoding_for_model(Settings.embed_model.model_name).encode
        self.init_token_counter(tokenizer)

    def init_token_counter(self, tokenizer):
        self.token_counter = TokenCountingHandler(tokenizer)
        Settings.callback_manager = CallbackManager([self.token_counter])

    def parse_json(self, cv_content):
        try: 
            return json.loads(cv_content)
        except json.decoder.JSONDecodeError:
            return json_repair.loads(cv_content)

    def generate_json(self, text, client: FunctionCallingLLM):
        formatted_prompt = prompt.format(json_format=json_format, cv_text=text)
        messages = [
            ChatMessage(
                role="user", 
                content=formatted_prompt,
            ),
        ]
        response: ChatResponse = client.chat(            
            messages=messages,
            response_format={"type": "json_object"},
        )

        try: 
            print(client._get_model_name())
            tokenizer = tiktoken.encoding_for_model(client._get_model_name())
            print(f"LLM Prompt Tokens: {len(tokenizer.encode(formatted_prompt))}")
            print(f"LLM Completion Tokens: {len(tokenizer.encode(response.message.content))}")
            print(f"Total LLM Token Count: {self.token_counter.prompt_llm_token_count + self.token_counter.completion_llm_token_count}")
        except Exception as error:
            print("Tokenizer not found in client. Skipping token count calculation. Error: ", error)

        cv_content = response.message.content
        cv_content_json = self.parse_json(cv_content)
        return cv_content_json
    
    def generate_json_w_parsed_json(self, resume_text, parsed_json_output, client: FunctionCallingLLM):
        formatted_prompt = prompt_w_parsed_json.format(json_format_schema=json_format, resume_text=resume_text, parsed_json_output=parsed_json_output)
        messages = [
            ChatMessage(
                role="user", 
                content=formatted_prompt,
            ),
        ]
        response: ChatResponse = client.chat(            
            messages=messages,
        )

        try: 
            print(client._get_model_name())
            tokenizer = tiktoken.encoding_for_model(client._get_model_name())
            print(f"LLM Prompt Tokens: {len(tokenizer.encode(formatted_prompt))}")
            print(f"LLM Completion Tokens: {len(tokenizer.encode(response.message.content))}")
            print(f"Total LLM Token Count: {self.token_counter.prompt_llm_token_count + self.token_counter.completion_llm_token_count}")
        except Exception as error:
            print("Tokenizer not found in client. Skipping token count calculation. Error: ", error)

        cv_content = response.message.content
        json_match = re.search(r'{[\s\S]*}', cv_content)
        if json_match:
            cv_json_str = json_match.group(0)
            cv_content_json = self.parse_json(cv_json_str)
            return cv_content_json            
        else:
            print("No JSON found in the text.")
            return None
    
    def improve_by_reiteration(self, text, client: FunctionCallingLLM):
        parsed_json = self.generate_json(text)
        reiteration_count = 2
        for i in range(reiteration_count):
            parsed_json = self.generate_json_w_parsed_json(text, json.dumps(parsed_json), client)

        return parsed_json
    
    def print_token_count_info(self):
        print(
            "Embedding Tokens: ",
            self.token_counter.total_embedding_token_count,
            "\n",
            "LLM Prompt Tokens: ",
            self.token_counter.prompt_llm_token_count,
            "\n",
            "LLM Completion Tokens: ",
            self.token_counter.completion_llm_token_count,
            "\n",
            "Total LLM Token Count: ",
            self.token_counter.total_llm_token_count,
            "\n",
        )
    
# json_format = """
# {
#     "name": str
#     "email": str
#     "phone": str
#     "address": str
#     "about": str
#     "skills": list<str>
#     "experience": list<{ "title": str, "company": str, "duration": str, "description": str }>
#     "education": list<{ "degree": str, "institution": str, "duration": str }>
#     "projects": list<{ "title": str, "description": str }>
#     "extra-activities": list<{ "title": str, "team": str?, "description": str }>
# }
# """

json_format = CVData.to_prompt()

empty_json = """
{
    "name": "",
    "email": "",
    "phone": "",
    "address": "",
    "about": "",
    "skills": [],
    "experience": [],
    "education": [],
    "projects": [ ],
    "extra-activities": []
}
"""

prompt = """
Get the CV data in the given json format from the CV text.
If there are any missing fields, fill them with None or empty strings.

Json format: 
{json_format}

CV text(parsed from CV images using OCR techniques):
{cv_text}
"""


prompt_w_parsed_json = """
Using the provided prompt, JSON format, resume text (these three are delimited by triple quotes), and the corresponding Parsed JSON Output , perform the following tasks:

1- Evaluate the Parsed JSON: Carefully review the Parsed JSON Output to ensure that each field is accurately and completely populated, based solely on the information explicitly or implicitly provided in the resume text.

2- Identify Issues: Identify any fields within the Parsed JSON that are incorrect, missing, or inadequately populated. Clearly specify each issue, refraining from making any assumptions beyond what is presented in the resume text.

3- Check Skills:

* Verify All Skills: Ensure that all relevant skills mentioned in the resume are included in the JSON.
* Add Missing Direct Skills: Add any direct skills from the resume that are missing in the JSON.
* Add Implied Skills: Include any skills that are implied based on the resume’s context.
* Remove Non-Relevant Skills: Remove any skills that do not pertain to the content of the resume.

4- Correct the JSON: Provide a corrected version of each problematic field by adjusting its content as needed, ensuring that corrections strictly align with the resume text.

5- Determine Seniority Level: Evaluate the seniority level indicated by the resume. Justify your assessment based on the content of the resume, avoiding any assumptions not supported by the resume text.

6- Final Output: Present the fully corrected JSON, preserving the original format, with all fields accurately and completely populated.

**PROMPT:**
\"\"\"Parse the fields from the following resume delimited by tripled quotes, and present them in the specified JSON format of which schema guide is given.
Place null wherever the information is not available.

**JSON Format Schema Guide**
{json_format_schema}

**Resume Text:**
\"\"\"{resume_text}\"\"\"

**PARSED JSON OUTPUT**
{parsed_json_output}
"""