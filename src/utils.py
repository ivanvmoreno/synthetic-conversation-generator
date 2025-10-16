import json
import os
from copy import deepcopy
from itertools import product
from typing import List, Optional, Set, Tuple

import yaml

from src.prompters.base import Prompter, Template
from src.prompters.seed import (
    Agent,
    Candidate,
    ConversationFlow,
    FlowStep,
    JobOpening,
    Role,
    Seed,
)


def load_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def save_yaml(data: dict, file_path: str):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Open the file and write the YAML data
    with open(file_path, "w") as yaml_file:
        yaml.dump(data, yaml_file)


def format_base_prompt(
    seed: Seed,
    base_template: str,
    template: Template,
) -> str:
    return base_template.format(
        **{
            name: val.format(**seed.model_dump())
            for name, val in template.model_dump().items()
        }
    ).strip()


def serialize_seed_conversation(
    seed: Seed, prompter: Prompter, include_system=True
) -> List[dict]:
    conversation_dict = seed.conversation.model_dump()
    # Add the fine-tuning system message prompt to beginning of conversation
    messages = (
        [
            {
                "content": format_base_prompt(
                    seed, prompter.fine_tuning_template, prompter.agent_template
                ),
                "role": "system",
            }
        ]
        if include_system
        else []
    )
    messages += [
        {
            "content": item["content"],
            "role": "assistant" if item["role"].value == Role.AGENT else "user",
        }
        for item in conversation_dict["messages"]
    ]
    return messages


def dump_conversation_jsonl(conversation: List[dict], file_path: str):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Open the file in append mode and write the JSONL data
    with open(file_path, "a") as jsonl_file:
        jsonl_file.write(json.dumps({"messages": conversation}) + "\n")


def generate_seeds(
    agents: List[Agent],
    candidates: List[Candidate],
    job_openings: List[JobOpening],
    conversation_flows: List[ConversationFlow],
) -> List[Seed]:
    """Generate all possible combinations of seeds given the input lists and conversation flow paths.

    Args:
        agents (List[Agent]): List of agents.
        candidates (List[Candidate]): List of candidates.
        job_openings (List[JobOpening]): List of job openings.
        conversation_flows (List[ConversationFlow]): List of conversation flow paths.

    Returns:
        List[Seed]: List of Seed instances.
    """

    seeds = []

    # Generate all possible combinations using itertools.product
    for agent, candidate, job_opening, conversation_flow_path in product(
        agents, candidates, job_openings, conversation_flows
    ):
        seed = Seed(
            agent=agent,
            candidate=candidate,
            job_opening=job_opening,
            conversation_flow=deepcopy(
                conversation_flow_path
            ),  # Deep copy to avoid shared state
        )
        seeds.append(seed)

    return seeds


def traverse_flow_with_decisions(
    conversation_flow: List["FlowStep"],
    start_id: str = "introduction",
    path: Optional[List[Tuple["FlowStep", Optional[str]]]] = None,
    paths: Optional[List[List[Tuple["FlowStep", Optional[str]]]]] = None,
    visited: Optional[Set[str]] = None,
) -> List["ConversationFlow"]:
    """Traverse a conversation flow and return all possible paths with decisions."""

    if path is None:
        path = []
    if paths is None:
        paths = []
    if visited is None:
        visited = set()

    # Locate the current node in the flow
    current_node = next(
        (node for node in conversation_flow if node.id == start_id), None
    )
    if not current_node:
        return paths

    # Avoid revisiting the same node within the same path
    if start_id in visited:
        return paths

    # Mark the current node as visited to avoid circular paths
    visited = visited.copy()  # Ensure that visited set is unique for each path
    visited.add(start_id)

    # Handle branching
    if current_node.is_branch:
        for decision, branch_id in current_node.next.items():
            branched_path = path.copy()
            branched_path.append((current_node, decision))

            # Recurse with the branch and decision
            traverse_flow_with_decisions(
                conversation_flow,
                branch_id,
                branched_path,
                paths,
                visited,
            )
    else:
        # Append the current step to the path
        path.append((current_node, current_node.decision))

        if current_node.is_end:
            paths.append(path.copy())  # Add the completed path to paths
        else:
            # Continue traversing the next step in the flow
            traverse_flow_with_decisions(
                conversation_flow, current_node.next, path, paths, visited
            )

        # Remove the current step after processing
        path.pop()

    # Convert paths to ConversationFlow objects, ensuring all decisions are preserved
    return [
        ConversationFlow(
            sequence=[
                (
                    step.model_copy(update={"decision": decision})
                    if decision
                    else step
                )
                for step, decision in path
            ]
        )
        for path in paths
    ]
