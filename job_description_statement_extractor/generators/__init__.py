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
from job_description_statement_extractor.response_model import StatementData

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
        formatted_prompt = prompt.format(json_format=json_format, text=text)
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
Review the provided job description and extract key statements according to the following format:

1. Job Information:
- Extract job title, company name, location, and job type
- Format as clear, complete statements

2. Must-Have Requirements:
- List all essential requirements and qualifications
- Include specific technical skills or certifications required

3. Nice-to-Have Requirements:
- List preferred but not mandatory qualifications
- Include desired skills or experiences

4. Responsibilities:
- List all job duties and responsibilities
- Keep original phrasing where possible

5. Required Skills:
- List all technical and professional skills required
- Include specific tools, technologies, or methodologies

6. Experience Required:
- Extract years of experience requirements
- Include specific domain experience requirements

7. Educational Requirements:
- List required degrees or certifications
- Include preferred institutions if mentioned

8. Additional Information:
- Include any other relevant requirements or preferences
- Add benefits, perks, or company culture information if mentioned

Job Description Text:
{text}

Format the output as a JSON object with the following structure:
{json_format}
"""