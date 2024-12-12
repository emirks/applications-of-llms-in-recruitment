from . import BaseGenerator
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
import os
from dotenv import load_dotenv
from typing import Optional, Dict, Any
import tiktoken

load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = api_key

client = OpenAI(model="gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY"))

class GPT4oGenerator(BaseGenerator):
    def __init__(self):
        super().__init__()
        self.client = client
    
    def generate_with_prompt(self, prompt: str, content: str, response_format: Optional[Dict] = None) -> str:
        messages = [
            ChatMessage(
                role="user", 
                content=f"{prompt}\n\nContent:\n{content}",
            ),
        ]
        
        response = self.client.chat(            
            messages=messages,
            response_format=response_format,
        )
        
        try: 
            print(self.client._get_model_name())
            tokenizer = tiktoken.encoding_for_model(self.client._get_model_name())
            print(f"LLM Prompt Tokens: {len(tokenizer.encode(prompt))}")
            print(f"LLM Completion Tokens: {len(tokenizer.encode(response.message.content))}")
            print(f"Total LLM Token Count: {self.token_counter.prompt_llm_token_count + self.token_counter.completion_llm_token_count}")
        except Exception as error:
            print("Tokenizer not found in client. Skipping token count calculation. Error: ", error)
            
        return response.message.content