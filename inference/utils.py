import json
import re
from typing import List, Union

import yaml


def get_id_from_list(lst: list[dict], id: str) -> dict:
    """Get a dict from a list of dicts by id."""
    res = next((i for i in lst if i["id"] == id), None)
    if res is None:
        raise ValueError(f"{id} not found in list")
    return res


def format_template(template: str, data: dict) -> str:
    """Replace string template variables with values from a dict."""
    for key in re.findall(r"{(.*?)}", template):
        if key in data:
            template = template.replace(f"{{{key}}}", data[key])
    return template


def format_message(role: str, content: str):
    """Format a single example into a string."""
    return f"{role.capitalize()}: {content}"


def load_parse_yaml(yaml_file: str) -> dict:
    """Open, read and deserialize a yaml file into a dict."""
    with open(yaml_file, "r") as f:
        yaml_data = yaml.safe_load(f)
    return yaml_data


def read_txt_file(txt_file: str) -> str:
    """Open, read and deserialize a txt file into a string."""
    with open(txt_file, "r") as f:
        txt_data = f.read()
    return txt_data


def json_to_dict(json_file: str) -> dict:
    """Open, read and deserialize a json file into a dict."""
    with open(json_file, "r") as f:
        json_data = json.load(f)
    return json_data


def omit(d: dict, excludes: list[str]) -> dict:
    keys = set(list(d.keys())) - set(excludes)
    return {k: d[k] for k in keys if k in d}
