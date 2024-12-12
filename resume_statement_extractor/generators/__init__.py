from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core import Settings
import tiktoken
from llama_index.core.llms import ChatMessage
import json
import json_repair
from typing import Dict, Any, Optional

# types
from llama_index.core.base.llms.types import ChatResponse
from llama_index.core.llms.function_calling import FunctionCallingLLM

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

    def parse_json(self, content: str) -> Dict[str, Any]:
        try: 
            return json.loads(content)
        except json.decoder.JSONDecodeError:
            return json_repair.loads(content)

    def generate_with_prompt(self, prompt: str, content: str, response_format: Optional[Dict] = None) -> str:
        """Base method to generate response with a given prompt"""
        raise NotImplementedError("Subclasses must implement generate_with_prompt")

    def generate_text(self, prompt: str, content: str) -> str:
        """Generate plain text response"""
        return self.generate_with_prompt(prompt, content)

    def generate_json(self, prompt: str, content: str) -> Dict[str, Any]:
        """Generate JSON response"""
        response = self.generate_with_prompt(
            prompt, 
            content,
            response_format={"type": "json_object"}
        )
        return self.parse_json(response)



    def _log_token_usage(self, client: FunctionCallingLLM, prompt: str, response: str):
        """Helper method to log token usage"""
        try: 
            print(client._get_model_name())
            tokenizer = tiktoken.encoding_for_model(client._get_model_name())
            print(f"LLM Prompt Tokens: {len(tokenizer.encode(prompt))}")
            print(f"LLM Completion Tokens: {len(tokenizer.encode(response))}")
            print(f"Total LLM Token Count: {self.token_counter.prompt_llm_token_count + self.token_counter.completion_llm_token_count}")
        except Exception as error:
            print("Tokenizer not found in client. Skipping token count calculation. Error: ", error)

