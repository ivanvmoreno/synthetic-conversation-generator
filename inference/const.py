from dotenv import load_dotenv
from utils import format_message, load_parse_yaml

load_dotenv(".env")

CONFIG_FILE = load_parse_yaml("../config/inference.yml")
MODEL_ID = CONFIG_FILE["model"]["id"]
MODEL_CONFIG = CONFIG_FILE["model"]["config"]
MODEL_SIGNATURE = CONFIG_FILE["model_signature"]
PROMPT_DATA = CONFIG_FILE["prompt_data"]
PROMPT_TPL = CONFIG_FILE["prompt"]
INPUT_EXAMPLES = CONFIG_FILE["input_examples"]
