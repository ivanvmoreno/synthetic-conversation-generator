from typing import Dict, List

import openai

from src.models.base import Model


class OpenAIModel(Model):

    def __init__(self, model_name="gpt-4o-mini"):
        super(OpenAIModel, self).__init__()
        self.model_name = model_name

    def generate(self, messages: List[Dict[str, str]], generation_params: dict) -> str:
        completion = openai.chat.completions.create(
            model=self.model_name, messages=messages, **generation_params
        )
        response = completion.choices[0].message.content
        return response.strip()
