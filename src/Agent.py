from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.Message import Message
from src.Probe import Probe


@dataclass
class BaseAgent:
    """Base class for all agents in the LLM debate simulation.

    This class represents an agent that maintains beliefs, interacts with neighbors,
    and uses neural probes to score beliefs. Agents can be either LLM-based or expert-based.

    Attributes:
        id: Unique identifier for the agent.
        model_name: Name of the model used by the agent (e.g., 'llama-base').
        cfg: Configuration dictionary for the agent.
        message: Message object for generating prompts.
        probe: Probe object for scoring beliefs.
        role: Optional role identifier (e.g., 'LLM', 'Participant').
        beliefs: Dictionary mapping round number to belief label.
        _belief_score: Dictionary mapping round number to belief score.
    """

    id: int
    model_name: str
    cfg: Dict
    message: Message
    probe: Probe
    role: Optional[str] = None

    beliefs: Dict[int, float] = field(default_factory=dict)
    _belief_score: Dict[int, float] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize agent after dataclass initialization."""
        self.message.set_role_and_context(role=self.role)
        self.message.set_query_names(
            user_name=self.cfg.get("model").get("query").get("user"),
            assistant_name=self.cfg.get("model").get("query").get("assistant"),
            system_name=self.cfg.get("model").get("query").get("system"),
        )
        self.probe.load_probe()

    def set_statement(self, s: str) -> None:
        """Set the statement to evaluate.

        Args:
            s: The statement to evaluate.
        """
        self.message.set_statement(s)

    def prepare_round(self, t: int, neighbor_view: Dict[int, float]) -> None:
        """Supply information for the current round.

        Args:
            t: The current round number.
            neighbor_view: Dictionary mapping neighbor IDs to their belief values.
        """
        self.message.update(t, neighbor_view)

    @property
    def current_message(self) -> List[Dict[str, str]]:
        """Get the current message as a chat template.

        Returns:
            List of dictionaries with 'role' and 'content' keys for chat format.
        """
        return self.message.as_chat_template()

    def set_belief(self, label: float, t: int, score: Optional[float] = None) -> None:
        """Set the belief for a given round.

        Args:
            label: Belief label value.
            t: Round number.
            score: Optional score associated with the belief.
        """
        if score is not None:
            self._belief_score[t] = float(score)
        self.beliefs[t] = float(label)

    def current_belief(self, t: int) -> Optional[float]:
        """Get the belief for a given round.

        Args:
            t: Round number.

        Returns:
            The belief value for round t, or None if not set.
        """
        return self.beliefs.get(t)

    def current_belief_score(self, t: int) -> Optional[float]:
        """Get the belief score for a given round.

        Args:
            t: Round number.

        Returns:
            The belief score for round t, or None if not set.
        """
        return self._belief_score.get(t)


@dataclass
class LLMAgent(BaseAgent):
    """Agent that uses a large language model for generating responses.

    This agent extends BaseAgent to provide LLM-specific functionality for
    querying language models during the simulation.
    """

    def generate(self, prompt: str, scheduler) -> str:
        """Query the LLM with a given prompt.

        Args:
            prompt: The prompt to send to the LLM.
            scheduler: InferenceScheduler object for managing model inference.

        Returns:
            The LLM's response as a string.
        """
        return scheduler.query_llm(self, prompt)


@dataclass
class ExpertAgent(BaseAgent):
    """Agent representing a domain expert with predefined knowledge.

    This agent extends BaseAgent to provide expert-specific functionality.
    Typically used as a baseline or for testing scenarios.
    """

    def generate(self, prompt: str, scheduler) -> str:
        """Query the expert agent with a given prompt.

        Args:
            prompt: The prompt to process.
            scheduler: InferenceScheduler object for managing inference.

        Returns:
            The expert agent's response as a string.
        """
        return scheduler.query_llm(self, prompt)
