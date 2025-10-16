from os import getenv

import mlflow
import runpod

MLFLOW_TRACKING_URI = getenv("MLFLOW_TRACKING_URI")
MODEL_ID = getenv("MODEL_ID")
MODEL_VERSION = getenv("MODEL_VERSION")
MODEL_TAG = getenv("MODEL_TAG")
MLFLOW_MODEL = (
    f"models:/{MODEL_ID}/{MODEL_VERSION}"
    if MODEL_VERSION
    else (
        f"models:/{MODEL_ID}@{MODEL_TAG}"
        if MODEL_TAG
        else f"models:/{MODEL_ID}@champion"
    )
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = mlflow.pyfunc.load_model(MLFLOW_MODEL)


def handler(event):
    job_input = event["input"]
    try:
        return model.predict(job_input["inputs"], params=job_input["params"])
    except Exception as e:
        print(e)
        return "Error processing input."


runpod.serverless.start({"handler": handler})
