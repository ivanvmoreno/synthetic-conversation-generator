import re
from operator import itemgetter
from typing import Callable, List, Mapping, Optional, Tuple

import mlflow
import pandas as pd
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
)
from mlflow.pyfunc import PythonModelContext


class AssistantModel(mlflow.pyfunc.PythonModel):
    @classmethod
    def _prepare_messages(cls, data) -> List[dict]:
        # https://github.com/mlflow/mlflow/blob/f643527f54eff4ac5d2b54ff8e8d24a5d6508848/mlflow/langchain/__init__.py#L556
        def _convert_ndarray_to_list(data):
            import numpy as np

            if isinstance(data, np.ndarray):
                return data.tolist()
            if isinstance(data, list):
                return [_convert_ndarray_to_list(d) for d in data]
            if isinstance(data, dict):
                return {k: _convert_ndarray_to_list(v) for k, v in data.items()}
            return data

        if isinstance(data, pd.DataFrame):
            return data.to_dict(orient="records")

        data = _convert_ndarray_to_list(data)
        if isinstance(data, list) and (
            all(isinstance(d, str) for d in data)
            or all(isinstance(d, dict) for d in data)
        ):
            return data
        raise mlflow.MlflowException.invalid_parameter_value(
            "Input must be a pandas DataFrame or a list of strings or a list of dictionaries."
        )

    def _format_template(self, template: str, data: dict) -> str:
        """Replace string template variables with values from a dict."""
        for key in re.findall(r"{(.*?)}", template):
            if key in data:
                template = template.replace(f"{{{key}}}", data[key])
        return template

    def _format_messages(self, messages: List[str], params: dict) -> List[dict]:
        formatted_messages = [
            {
                "content": self._format_template(
                    self.prompt, {**self.prompt_data, **params}
                ),
                "role": "system",
            }
        ]

        for message in messages:
            if isinstance(message, dict):
                formatted_messages.append(message)
            else:
                formatted_messages.append({"content": message, "role": "user"})

        return formatted_messages

    def _get_llm(
        self,
        tf_model,
        model_params,
        quant,
        context_window: Optional[int] = 2048,
        stop_list: Optional[List[str] | None] = None,
        gpus=1,
        gpu_mem=0.9,
    ):
        """Instantiates an inference server and returns a wrapper."""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=tf_model,
            load_in_4bit=True,
        )
        self.model = model
        self.tokenizer = get_chat_template(
            tokenizer,
            chat_template="chatml",
            map_eos_token=True,
        )
        FastLanguageModel.for_inference(model)

    def _get_assistant_message(self, output: str) -> str:
        assistant_messages = re.findall(
            r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", output, re.DOTALL
        )
        if assistant_messages:
            return assistant_messages[-1].strip()
        return ""

    def __init__(self, prompt: str, prompt_data: dict):
        self.prompt_data = prompt_data
        self.prompt = prompt

    def load_context(self, context: PythonModelContext):
        gpus = torch.cuda.device_count() or 1
        model_config = context.model_config
        stop_list = (
            context.model_config["stop_list"]
            if "stop_list" in model_config
            else None
        )
        self._get_llm(
            tf_model=model_config["hf_model_id"],
            model_params=model_config["model_params"],
            quant=model_config["quantization"],
            context_window=model_config["context_window"],
            stop_list=stop_list,
            gpus=gpus,
        )

    def predict(self, context, model_input, params):
        messages = self._prepare_messages(model_input)[0]["messages"]
        messages = self._format_messages(messages, params)
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")
        outputs = self.model.generate(
            input_ids=inputs, max_new_tokens=256, use_cache=True
        )
        decoded = self.tokenizer.batch_decode(outputs)[0]
        return self._get_assistant_message(decoded)
