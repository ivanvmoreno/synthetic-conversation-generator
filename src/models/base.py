from typing import Dict, List


class Model:
    def generate(self, messages: List[Dict[str, str]], generation_params: dict) -> str:
        raise NotImplemented
