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
    
json_format = StatementData.to_prompt()

prompt = """
Extract statements from the CV data in the given JSON format.
For each section, extract relevant information and format according to the schema.
If information is not available, use null or empty values.

Required Sections:
1. Personal Information:
   - Name, DOB, Gender, Location, Position, Email
   - Format as clear, concise statements

2. Education Information:
   - Educational history and qualifications
   - Include institution, degree, timeframe

3. Certificates Information:
   - Professional certifications and qualifications
   - Include certification name and issuing body

4. Personality Information:
   - Self-described traits and characteristics
   - Professional soft skills

5. Skills and Experience:
   - Technical and professional skills
   - Years of experience and proficiency level
   - Format as "Skill: <name>, Experience: <description>, Years: <value>, Level: <value>"

Json format: 
{json_format}

CV text:
{cv_text}
"""
