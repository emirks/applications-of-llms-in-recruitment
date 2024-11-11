from . import BaseParser

class TextParser(BaseParser):
    def parse_text(self, file_path: str, text_content: list) -> str:
        # Since we're dealing with text files, just return the first element
        return text_content[0] 