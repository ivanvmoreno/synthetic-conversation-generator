from dataclasses import dataclass
from typing import Dict, Optional

from pydantic import BaseModel

from src.prompters.seed import ActionType, FlowStep, Role, Seed, StepAction


class Template(BaseModel):
    system_message: str
    description_template: str
    context_template: str


class Prompter:
    def __init__(
        self,
        base_template: str,
        fine_tuning_template: str,
        agent_template: Template,
        candidate_template: Template,
    ):
        self.base_template = base_template
        self.fine_tuning_template = fine_tuning_template
        self.agent_template = agent_template
        self.candidate_template = candidate_template

    def get_prompt(self, seed: Seed) -> str:
        """
        Generate system message prompt for the current turn

        Args:
            seed (Seed): Seed object containing the current state of the conversation

        Returns:
            str: System message prompt for the current turn
        """
        turn = seed.conversation_flow.current_turn
        step = turn.step
        action = step.get_role_action(turn.role)

        template = (
            self.agent_template
            if turn.role == Role.AGENT
            else self.candidate_template
        )

        prompt = self.base_template.format(
            stage_description=action.description,
            stage_example=self.format_action(step, action, seed),
            **{
                name: val.format(**seed.model_dump())
                for name, val in template.model_dump().items()
            },
        )
        return prompt.strip()

    def format_action(
        self, step: FlowStep, action: StepAction, seed: Seed
    ) -> str:
        res = (
            action.example.format(**seed.model_dump())
            if action.action_type in {ActionType.TEMPLATE, ActionType.MESSAGE}
            else action.examples[step.decision].format(**seed.model_dump())
        )
        return res.strip()
