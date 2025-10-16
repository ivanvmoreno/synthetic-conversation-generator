import os
import json

from dotenv import load_dotenv
import mlflow
from mlflow.models import infer_signature

from model import AssistantModel
from const import (
    MODEL_ID,
    MODEL_CONFIG,
    MODEL_SIGNATURE,
    INPUT_EXAMPLES,
    PROMPT_DATA,
    PROMPT_TPL,
)

load_dotenv(".env")


if __name__ == "__main__":
    signature = infer_signature(
        model_input=MODEL_SIGNATURE["input"], params=MODEL_SIGNATURE["params"]
    )
    model = AssistantModel(PROMPT_TPL, PROMPT_DATA)
    # log model on tracking server
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    if MLFLOW_TRACKING_URI is None:
        raise ValueError("MLFLOW_TRACKING_URI is not set")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MODEL_ID)
    mlflow_client = mlflow.tracking.MlflowClient()
    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            artifact_path="model",
            registered_model_name=MODEL_ID,
            python_model=model,
            pip_requirements="./requirements.txt",
            model_config=MODEL_CONFIG,
            signature=signature,
            code_paths=["model.py"],
        )
        mlflow.log_params(
            {
                "instruction_tpl": PROMPT_TPL,
                "instruction_tpl_data": json.dumps(PROMPT_DATA),
                **MODEL_CONFIG,
            }
        )
        registered_model = mlflow_client.search_model_versions(
            filter_string=f"name='{MODEL_ID}'",
            order_by=["creation_timestamp DESC"],
        ).to_list()[0]
        # support for latest model version - alias
        mlflow_client.delete_registered_model_alias(MODEL_ID, "last")
        mlflow_client.set_registered_model_alias(
            MODEL_ID, "last", registered_model.version
        )
        mlflow.end_run()
