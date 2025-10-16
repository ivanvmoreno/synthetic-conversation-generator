from typing import Any, Dict, List

from src.prompters.base import Prompter
from src.prompters.seed import Role, Seed


class OpenAIPrompter(Prompter):
    def get_prompt(self, seed: Seed) -> List[Dict[str, str]]:
        conversation_history = self.format_conversation_history(
            seed.conversation.messages, seed.conversation_flow.current_role
        )
        # Add system message prompt
        system_prompt = super().get_prompt(seed)
        conversation_history.insert(
            0, {"role": "system", "content": system_prompt}
        )

        return conversation_history

    def format_conversation_history(
        self, history: List[Any], current_role: Role
    ) -> List[Dict[str, str]]:
        formatted_history = [
            {
                "role": "assistant" if turn.role == current_role else "user",
                "content": turn.content,
            }
            for turn in history
        ]
        return formatted_history
