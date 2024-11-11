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
from resume_statement_extractor.response_model import StatementData

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

json_format = StatementData.to_prompt()

prompt = """
Extract statements from the CV data in the given JSON format.
For each section, extract relevant information and format according to the schema.
If information is not available, use null or empty values.

Json format: 
{json_format}

CV text:
{cv_text}
"""

prompt_w_parsed_json = """
Review the provided resume text and extract key statements according to the following format:

1. Personal Information:
- Extract basic details like name, DOB, gender, location, position, and email
- Ensure accuracy and completeness

2. Education:
- List all educational qualifications as clear statements
- Include institution, degree, and timeframe if available

3. Certifications:
- Extract all professional certifications
- Include certification name and issuing body

4. Personality Traits:
- Identify self-described personality traits
- Extract relevant soft skills and characteristics

5. Skills:
- List technical and professional skills
- Include years of experience and proficiency level where available
- Format as "Skill: <name>, Associated Experience: <full description text>, Years: <value>, Proficiency: <value>"
* Verify All Skills: Ensure that all relevant skills mentioned in the resume are included in the JSON.
JSON Format Schema:
{json_format_schema}
* Remove Non-Relevant Skills: Remove any skills that do not pertain to the content of the resume.
Resume Text:
{resume_text}

Current Parsed JSON:
{parsed_json_output}
"""