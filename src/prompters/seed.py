from enum import Enum
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Role(str, Enum):
    AGENT = "agent"
    CANDIDATE = "candidate"

    def __str__(self):
        return self.value


class Experience(BaseModel):
    company: str
    position: str
    description: str


class Candidate(BaseModel):
    first_name: str
    last_name: str
    experiences: List[Experience]


class Agent(BaseModel):
    first_name: str
    last_name: str
    company_name: str
    company_info: str
    career_site_url: str


class JobOpening(BaseModel):
    title: str
    description: str
    requirements: List[str]


class ActionType(str, Enum):
    MESSAGE = "message"
    RESPONSE = "response"
    TEMPLATE = "template"


class ResponseType(str, Enum):
    TEMPLATE = "template"
    MESSAGE = "message"


class StepAction(BaseModel):
    role: Role
    action_type: ActionType
    description: str
    response_type: Optional[ResponseType] = None
    example: Optional[str] = None
    examples: Optional[Dict[Literal["yes", "no"], str]] = None

    @model_validator(mode="before")
    def validate_examples(cls, values):
        example, examples = values.get("example"), values.get("examples")
        if example and examples:
            raise ValueError(
                "StepAction can have either 'example' or 'examples', not both."
            )
        if (
            not example
            and not examples
            and values.get("action_type") != ActionType.TEMPLATE
        ):
            raise ValueError(
                "StepAction must have either 'example' or 'examples'."
            )
        return values


class FlowStep(BaseModel):
    """
    A class representing a flow step with various attributes.

    Attributes:
        id (str): The unique identifier of the flow step.
        initiator (Role): The role that initiates the flow step.
        actions (List[Dict[str, str]]): A list of actions to perform in the flow step.
        is_branch (bool): Flag indicating if the flow step is a branch (default is False).
        is_end (bool): Flag indicating if the flow step is the end of the conversation (default is False).
        next (Optional[Union[str, Dict[str, str]]]): The next step in the conversation flow (default is None).
        decision (Optional[Literal["yes", "no"]]): The decision to make for branch steps (default is None).
    """

    id: str
    initiator: Role
    actions: List[StepAction]
    is_branch: bool = False
    is_start: bool = False
    is_end: bool = False
    next: Optional[Union[str, Dict[Literal["yes", "no"], str]]] = None
    decision: Optional[Literal["yes", "no"]] = None

    @model_validator(mode="before")
    def validate_flow_step(cls, values):
        is_branch = values.get("is_branch")
        is_end = values.get("is_end")
        next_step = values.get("next")
        decision = values.get("decision")

        if is_end and next_step:
            raise ValueError("End steps should not have a next step.")
        if not is_end and not next_step:
            raise ValueError("Non-end steps should have a next step.")
        if not is_branch and decision:
            raise ValueError("Decision is only allowed for branch steps.")
        if is_branch and not isinstance(next_step, dict):
            raise ValueError(
                "Next step should be a dictionary for branch steps."
            )
        return values

    def get_role_action(self, role: Role) -> StepAction:
        """
        Get the action to perform based on the role.

        Args:
            role (Role): The role to get the action for.

        Returns:
            StepAction: The action to perform.
        """
        for action in self.actions:
            if action.role == role:
                return action
        raise ValueError(f"No action found for role: {role}")


class ConversationTurn(BaseModel):
    role: Role
    step: FlowStep
    content: Optional[str] = None

    model_config = ConfigDict(
        exclude={"step"},
    )


class ConversationFlow(BaseModel):
    sequence: List[FlowStep]
    current_step: FlowStep = Field(default_factory=dict)
    current_role: Role = Field(default_factory=dict)
    completed_actions: List[Role] = Field(default_factory=list)

    def __init__(self, **data):
        super().__init__(**data)
        self.current_step = next(
            (step for step in self.sequence if step.is_start), None
        )
        if not self.current_step:
            raise ValueError("No start step found in the sequence.")
        self.current_role = self.current_step.initiator
        self.completed_actions = []

    @property
    def current_turn(self) -> Optional[ConversationTurn]:
        return ConversationTurn(role=self.current_role, step=self.current_step)

    def _swap_roles(self):
        # Swap roles if there are multiple actions and not all are completed
        if len(self.current_step.actions) > 1:
            self.current_role = (
                Role.CANDIDATE
                if self.current_role == Role.AGENT
                else Role.AGENT
            )

    def mark_action_completed(self, role: Role):
        if role not in self.completed_actions:
            self.completed_actions.append(role)

    def all_actions_completed(self) -> bool:
        required_roles = {action.role for action in self.current_step.actions}
        return required_roles.issubset(self.completed_actions)

    def transition(self) -> Union[ConversationTurn, None]:
        """
        Determine whether to swap turns or advance to the next step.
        Returns the next ConversationTurn, or None if the conversation is complete.
        """
        # Mark current role's action as completed
        self.mark_action_completed(self.current_role)

        # If all actions for the current step are completed, move to the next step
        if self.all_actions_completed():
            if self.current_step.is_end:
                return None

            # Handle branch logic
            if self.current_step.is_branch:
                if self.current_step.decision is None:
                    raise ValueError(
                        "Decision is required for branch transitions."
                    )
                next_step_id = self.current_step.next[
                    self.current_step.decision
                ]
            else:
                next_step_id = self.current_step.next

            # Transition to the next step
            self.current_step = next(
                (step for step in self.sequence if step.id == next_step_id),
                None,
            )
            if not self.current_step:
                raise ValueError(
                    f"Step with id '{next_step_id}' not found in the sequence."
                )

            # Reset completed actions and set the role to the new initiator
            self.completed_actions = []
            self.current_role = self.current_step.initiator
        else:
            # If not all actions are completed, swap roles
            self._swap_roles()

        return self.current_turn

    def __str__(self) -> str:
        return " -> ".join([step.id for step in self.sequence])


class Conversation(BaseModel):
    """
    A class representing a conversation with various attributes.

    Attributes:
        messages (List[ConversationTurn]): A list of conversation turns.
    """

    messages: List[ConversationTurn] = Field(default_factory=list)

    def append(self, turn: ConversationTurn):
        self.messages.append(turn)

    def __str__(self) -> str:
        return "\n".join(
            [
                f"{turn.role.value.capitalize()}: {turn.content}"
                for turn in self.messages
            ]
        )


class Seed(BaseModel):
    """
    A class representing a conversation seed with various attributes.

    Attributes:
        agent (Agent): The agent initiating the conversation.
        candidate (Candidate): The candidate participating in the conversation.
        job_opening (JobOpening): The job opening being discussed.
        conversation_flow (ConversationFlow): The conversation flow to follow.
        conversation (List[ConversationTurn]): The conversation history (default is None).
    """

    agent: Agent
    candidate: Candidate
    job_opening: JobOpening
    conversation_flow: ConversationFlow
    conversation: Conversation = Field(default_factory=Conversation)
