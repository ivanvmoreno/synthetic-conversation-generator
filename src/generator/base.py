import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List

from tenacity import retry, stop_after_attempt, wait_random_exponential

from src.models.base import Model
from src.prompters.base import Prompter
from src.prompters.seed import ActionType, ConversationTurn, ResponseType, Seed
from src.utils import dump_conversation_jsonl, serialize_seed_conversation


class Generator:
    def __init__(self, prompter: Prompter, model: Model):
        """
        :param prompter: The Prompter instance used to generate prompts.
        :param model: The Model instance used to generate responses.
        """
        self.prompter = prompter
        self.model = model
        self.executor = ThreadPoolExecutor()

    @retry(
        wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6)
    )
    async def _rate_limited_generate(self, prompt, generation_params):
        """
        Generate a response from the model with exponential backoff in case of failure.
        """
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            self.executor, self.model.generate, prompt, generation_params
        )
        return response

    async def _process_seed(
        self, seed: Seed, generation_params=None, save_options=None
    ) -> Seed:
        """
        Process a single seed by generating responses for each turn in the conversation.
        :param seed: The Seed instance to process.
        :param generation_params: Optional parameters for the model generation.
        :param save_options: Optional parameters for saving the conversation.
        :return: The processed Seed instance.
        """
        if generation_params is None:
            generation_params = {}

        while True:
            turn = seed.conversation_flow.current_turn
            if turn is None:
                break  # End of the conversation

            action = turn.step.get_role_action(turn.role)
            if (
                action.action_type == ActionType.TEMPLATE
                or action.response_type == ResponseType.TEMPLATE
            ):
                response = self.prompter.format_action(turn.step, action, seed)
            else:
                prompt = self.prompter.get_prompt(seed)
                response = await self._rate_limited_generate(
                    prompt, generation_params
                )

            seed.conversation.append(
                ConversationTurn(
                    role=turn.role, content=response, step=turn.step
                )
            )

            if seed.conversation_flow.current_step.is_end:
                break
            seed.conversation_flow.transition()

        if save_options is not None:
            self._save_conversation(seed, save_options["timestamp"])

        return seed

    def _save_conversation(self, seed: Seed, timestamp: str, path="output"):
        """
        Save the processed seed's conversation to a JSONL file.
        """
        dump_path = (
            f"{path}/conversations_{self.model.model_name}_{timestamp}.jsonl"
        )
        conversation = serialize_seed_conversation(seed, self.prompter)
        dump_conversation_jsonl(conversation, dump_path)

    async def generate(
        self,
        seeds: List[Seed],
        generation_params=None,
        save=True,
        save_path="output",
    ) -> List[Seed]:
        """
        Generate responses for a list of seeds.

        Args:
            seeds: List of Seed instances to process.
            generation_params: Optional parameters for the model generation.
            save: Whether to save the conversations to disk.
            save_path: The path to save the conversation files.

        Returns:
            List of processed Seed instances.
        """
        if generation_params is None:
            generation_params = {}

        now = datetime.now()
        timestamp = now.strftime("%d-%m-%Y_%H-%M")

        tasks = []
        for seed in seeds:
            if not save:
                tasks.append(self._process_seed(seed, generation_params))
            else:
                tasks.append(
                    self._process_seed(
                        seed,
                        generation_params,
                        {"timestamp": timestamp, "path": save_path},
                    )
                )

        await asyncio.gather(*tasks)
