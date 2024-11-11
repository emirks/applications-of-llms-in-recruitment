import json
class BaseProcessor:
    def process(self, content) -> json:
        raise NotImplementedError("BaseProcessor subclasses should implement 'process' function")
