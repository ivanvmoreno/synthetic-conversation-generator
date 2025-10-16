import asyncio
import os

import openai
from dotenv import load_dotenv

from src.generator.base import Generator
from src.models.openai_model import OpenAIModel
from src.prompters.base import Template
from src.prompters.openai import OpenAIPrompter
from src.prompters.seed import Agent, Candidate, FlowStep, JobOpening
from src.utils import generate_seeds, load_yaml, traverse_flow_with_decisions

load_dotenv()  # Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_KEY")

openai.api_key = OPENAI_API_KEY
model = OpenAIModel(model_name="gpt-4o-mini")

openai_generation_params = {
    "temperature": 1.0,
    "top_p": 1.0,
    "n": 1,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}


async def main():
    # Load configuration files
    templates_raw = load_yaml("config/templates.yml")
    templates = {
        "base": templates_raw["base"],
        "fine_tuning": templates_raw["fine_tuning"],
        "agent": Template.model_validate(templates_raw["agent"]),
        "candidate": Template.model_validate(templates_raw["candidate"]),
    }

    seeds_raw = load_yaml("config/seeds.yml")
    seeds = {
        "agents": [
            Agent.model_validate(agent) for agent in seeds_raw["agents"]
        ],
        "candidates": [
            Candidate.model_validate(candidate)
            for candidate in seeds_raw["candidates"]
        ],
        "job_openings": [
            JobOpening.model_validate(job_opening)
            for job_opening in seeds_raw["job_openings"]
        ],
    }

    flow_steps_raw = load_yaml("config/conversation_flow.yml")
    flow_steps = [
        FlowStep.model_validate(flow_step) for flow_step in flow_steps_raw
    ]

    # Traverse conversation flow with decisions
    conversation_flows = traverse_flow_with_decisions(flow_steps)

    # Generate seeds
    seeds = generate_seeds(
        agents=seeds["agents"],
        candidates=seeds["candidates"],
        job_openings=seeds["job_openings"],
        conversation_flows=conversation_flows,
    )

    # Initialize prompter and generator
    prompter = OpenAIPrompter(
        fine_tuning_template=templates["fine_tuning"],
        base_template=templates["base"],
        agent_template=templates["agent"],
        candidate_template=templates["candidate"],
    )
    generator = Generator(prompter=prompter, model=model)

    # Generate conversations
    await generator.generate(
        seeds=seeds, generation_params=openai_generation_params, save=True
    )


if __name__ == "__main__":
    asyncio.run(main())
