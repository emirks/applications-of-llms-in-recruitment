from . import BaseParser

class TextParser(BaseParser):
    def parse_text(self, file_path: str, text_content: str) -> str:
        # For text files, just return the content directly
        return text_content 