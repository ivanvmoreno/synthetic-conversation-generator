import os
import tempfile

from dotenv import load_dotenv
import mlflow
from mlflow.models import infer_signature
from utils import load_parse_yaml

from model import AssistantModel
from const import (
    MODEL_CONFIG,
    MODEL_SIGNATURE,
    INPUT_EXAMPLES,
    PROMPT_DATA,
    PROMPT_TPL,
)

load_dotenv()

if __name__ == "__main__":
    # test model locally (without logging it)
    signature = infer_signature(
        model_input=MODEL_SIGNATURE["input"], params=MODEL_SIGNATURE["params"]
    )
    model = AssistantModel(PROMPT_TPL, PROMPT_DATA)
    INPUT_TEST = load_parse_yaml("./input_test.yaml")
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "model")
        mlflow.pyfunc.save_model(
            pip_requirements="./requirements.txt",
            model_config=MODEL_CONFIG,
            input_example=INPUT_EXAMPLES,
            code_paths=["model.py"],
            signature=signature,
            path=model_path,
            python_model=model,
        )
        model = mlflow.pyfunc.load_model(model_path)
        print(
            model.predict(
                [{"messages": INPUT_TEST["messages"]}],
                params=INPUT_TEST["params"],
            )
        )
