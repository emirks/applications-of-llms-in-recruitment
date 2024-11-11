from . import BaseReader
import logging

logger = logging.getLogger(__name__)

class TextReader(BaseReader):
    def read_image(self, file_path: str):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            return [text]  # Return as list to maintain compatibility with image reader interface
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {e}")
            return None
