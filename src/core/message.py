from __future__ import annotations

import logging
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from jinja2 import Environment, Template, TemplateError

log = logging.getLogger("Message")


def clean_str(s: str) -> str:
    """Clean string by removing extra spaces and newlines.

    Args:
        s: input string

    Returns:
        cleaned string
    """
    s = s.replace("  ", " ")
    s = s.replace("..", ".")
    return s.strip()


@dataclass
class Message:
    """Message objects contains all the information needed to create a chat-like prompt.

    Methods (used by Agent and InferenceScheduler):
        update(t, neighbor_view): Update and store the message for round t with neighbor views.
        as_chat_template(): Get the complete chat template for LLM querying

    Note:
        To change the templates go to the config/prompt/ folder.
    """

    cfg: dict  # global configuration dictionary
    role: str = ""
    statement: str = ""
    context_prompt: str = ""

    # per-round log: prompt + reply
    history: dict[int, dict[str, str | None]] = field(default_factory=dict)

    # Templates and jinja2 engine
    _engine: str = field(init=False, repr=False, default="jinja2")
    # Compiled Jinja2 templates (if using jinja2)
    _jinja_templates: dict[str, Template | None] = field(
        init=False, default_factory=dict
    )
    # Raw template strings (for fallback)
    _raw_templates: dict[str, str] = field(init=False, default_factory=dict)

    # Chat Role (varies per LLM's API)
    _system_name: str = field(init=False, default="system")
    _user_name: str = field(init=False, default="user")
    _assistant_name: str = field(init=False, default="assistant")

    # Neighbor aggregation
    _method: str | None = field(init=False, repr=False)
    _probe: str | None = field(init=False, repr=False)

    # Random State
    _rng: random.Random = field(init=False, repr=False)
    # Flags
    _is_role_set: bool = field(init=False, default=False)
    _is_statement_set: bool = field(init=False, default=False)
    _is_query_set: bool = field(init=False, default=False)

    _agent_roles: dict[int, str] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self):
        """Initialize templates and configuration after dataclass init."""
        # get prompt configuration
        # load prompt config, see configs/prompt/
        p = self.cfg.get("prompt", {})
        self._engine = p.get("engine", "jinja2")  # template engine

        self._load_templates(p)  # load templates from config/prompt

        # Aggregation
        agg = p.get("aggregation", {}) or {}
        self._method = agg.get("method", "count")  # the fallback is 'count'
        self._probe = self.cfg.get("probe", {}).get("name", "sawmil")
        seed = self.cfg.get("seed", None)
        self._rng = random.Random(seed) if seed is not None else random.Random()
        catalog = self.cfg.get("agents", {}).get("catalog_used", [])
        self._agent_roles = {}
        for agent in catalog:
            agent_id = agent.get("id")
            role = agent.get("role")
            if agent_id is None:
                raise ValueError("Agent entry is missing an 'id' in the catalog")
            if not isinstance(role, str) or not role.strip():
                raise ValueError(
                    f"Agent with id={agent_id} has an empty or missing role in the catalog"
                )
            self._agent_roles[agent_id] = role.strip()

    def _load_templates(self, prompt_cfg: dict) -> None:
        """Load and compile Jinja2 templates from the configuration.

        Args:
            prompt_cfg: prompt configuration dictionary
        """
        t = prompt_cfg.get("template", {})

        # Store templates
        self._raw_templates["system"] = (
            t.get("system") or t.get("context", "")
        ).strip()

        u = t.get("user", {}) or {}
        self._raw_templates["intro"] = (u.get("intro") or "").strip()
        self._raw_templates["agree"] = (u.get("agree") or "").strip()
        self._raw_templates["disagree"] = (u.get("disagree") or "").strip()
        self._raw_templates["neutral"] = (u.get("neutral") or "").strip()
        self._raw_templates["instruction"] = (
            u.get("instruction", "What do you think about this statement?")
        ).strip()
        self._raw_templates["agent"] = (t.get("agent") or "").strip()

        if self._engine == "jinja2":
            env = Environment()
            for key, template_text in self._raw_templates.items():
                try:
                    template = env.from_string(template_text)
                    self._jinja_templates[key] = template
                except TemplateError as e:
                    log.error(f"Error compiling Jinja2 template for '{key}': {e}")
                    self._jinja_templates[key] = None

    def _render_template(self, template_key: str, **kwargs: Any) -> str:
        """Render a template with the given variables.

        Args:
            template_key: Key identifying the template to render.
            **kwargs: Template variables to render.

        Returns:
            Rendered template string.
        """
        if self._engine == "jinja2" and template_key in self._jinja_templates:
            template = self._jinja_templates[template_key]
            if template:
                try:
                    return template.render(**kwargs).strip()
                except Exception as e:
                    log.warning(f"Jinja2 render failed for '{template_key}': {e}")

        # Fallback to simple string replacement
        template_str = self._raw_templates.get(template_key, "")
        if not template_str:
            return ""

        return template_str.strip()

    # SETTERS
    def set_role_and_context(
        self, role: str | None, sys_template_override: str | None = None
    ) -> bool:
        """Store the system/context prompt inside the message object.

        Args:
            role: the LLM's role
            sys_template_override: the template to use to format the message.

        Returns:
            True if role was set successfully.
        """
        self.role = role or "LLM"

        if sys_template_override:
            # Use override template directly
            self.context_prompt = clean_str(
                sys_template_override.replace("{{role}}", self.role)
            )
        else:
            # Use configured template
            self.context_prompt = clean_str(
                self._render_template("system", role=self.role, probe=self._probe)
            )

        self._is_role_set = True
        return self._is_role_set

    def set_statement(self, s: str) -> bool:
        """Set the statement used to evaluate beliefs.

        Args:
            s: statement string

        Returns:
            True if statement was set successfully.
        """
        assert s is not None, "Statement must be a non-empty string."
        self.statement = s
        self._is_statement_set = True
        return self._is_statement_set

    def set_query_names(
        self, user_name: str, assistant_name: str, system_name: str
    ) -> bool:
        """Set the names used in the chat prompt (various model APIs use different conventions).

        Args:
            user_name: name of the user role
            assistant_name: name of the assistant role
            system_name: name of the system role

        Returns:
            True if query names were set successfully.
        """
        assert user_name and assistant_name and system_name, (
            "All names must be non-empty strings."
        )
        self._user_name = user_name
        self._assistant_name = assistant_name
        self._system_name = system_name
        self._is_query_set = True
        return self._is_query_set

    @property
    def _is_ready(self) -> bool:
        """Check if the message is ready to build prompts.

        Returns:
            True if ready (role, statement, and query names are set), False otherwise.
        """
        return self._is_role_set and self._is_statement_set and self._is_query_set

    @property
    def agent_prompt(self) -> str:
        """Get the agent-specific prompt.

        Returns:
            the agent prompt text
        """
        return self._build_agent_prompt()

    def update(self, t: int, neighbor_view: dict[int, float]) -> str:
        """Build & store the user prompt for round t if not already present.

        Returns the user prompt string.

        Args:
            t: round t
            neighbor_view: the neighbor beliefs

        Returns:
            the prompt for the LLM
        """
        assert self._is_ready, (
            "Role, statement, and query names must be set before building prompts."
        )

        if t in self.history and self.history[t].get("prompt"):
            prompt_val = self.history[t]["prompt"]
            if prompt_val is not None:
                return prompt_val

        if t == 0 or neighbor_view is None:
            prompt = self._build_init_user_prompt()
        else:
            prompt = self._build_user_prompt(neighbor_view)

        self.history.setdefault(t, {})["prompt"] = prompt
        return clean_str(prompt)

    def _build_init_user_prompt(self) -> str:
        """Build the initial prompt to send to the LLM.

        Returns:
            the prompt text
        """
        intro = self._render_template("intro", statement=self.statement)
        instruction = self._render_template(
            "instruction", probe=self._probe, statement=self.statement
        )
        return clean_str(f"{intro} {instruction}")

    def _build_agent_prompt(self) -> str:
        """Build the agent-specific prompt to send to the LLM.

        Returns:
            the prompt text
        """
        return self._render_template("agent", probe=self._probe)

    def _build_user_prompt(
        self, neighbor_view: dict[int, float], shuffle: bool = True
    ) -> str:
        """Build the prompt to send to the LLM based on the neighbor beliefs.

        Args:
            neighbor_view: dictionary of neighbor beliefs
            shuffle: whether to shuffle the order of the neighbor summaries

        Returns:
            the prompt text
        """
        intro = self._render_template("intro", statement=self.statement)

        # Partition neighbors
        agree_ids, disagree_ids, neutral_ids = self._partition(neighbor_view)

        # Build stance summaries
        stance_parts = []

        if agree_ids:
            part = self._build_stance_summary("agree", agree_ids)
            if part:
                stance_parts.append(part)

        if disagree_ids:
            part = self._build_stance_summary("disagree", disagree_ids)
            if part:
                stance_parts.append(part)

        if neutral_ids:
            part = self._build_stance_summary("neutral", neutral_ids)
            if part:
                stance_parts.append(part)

        # Shuffle stance parts if requested
        if shuffle and stance_parts:
            self._rng.shuffle(stance_parts)

        # Build final prompt
        instruction = self._render_template(
            "instruction", probe=self._probe, statement=self.statement
        )
        parts = [intro, *stance_parts, instruction]

        return clean_str(" ".join(p for p in parts if p))

    def _build_stance_summary(self, stance: str, agent_ids: list[str]) -> str:
        """Build summary for a specific stance.

        Args:
            stance: stance type ('agree', 'disagree', 'neutral')
            agent_ids: list of agent IDs with this stance

        Returns:
            formatted stance summary string
        """
        if not agent_ids:
            return ""

        if self._method == "count":
            # Count-based aggregation
            return self._render_template(
                stance, n=len(agent_ids), statement=self.statement
            )
        # List-based aggregation with duplicate handling
        agent_labels = [self._get_role(int(i)) for i in agent_ids]

        # Check if we have duplicates that need grouping
        if len(agent_labels) != len(set(agent_labels)):
            # We have duplicates - format with counts
            return self._format_grouped_stance(stance, agent_labels)
        # No duplicates - use template as-is
        return self._render_template(
            stance, agents=agent_labels, statement=self.statement
        )

    def _format_grouped_stance(self, stance: str, roles: list[str]) -> str:
        """Format stance summary when roles have duplicates.

        Groups roles by count and creates a grammatically correct summary.

        Args:
            stance: stance type ('agree', 'disagree', 'neutral')
            roles: list of role strings (may contain duplicates)

        Returns:
            formatted stance summary string
        """
        role_counts = Counter(roles)
        formatted_parts = []

        for role, count in role_counts.items():
            if count == 1:
                formatted_parts.append(role)
            else:
                plural_role = self._pluralize_role(role)
                formatted_parts.append(f"{count} {plural_role}")

        # Format the list grammatically
        if len(formatted_parts) == 1:
            agents_str = formatted_parts[0]
        elif len(formatted_parts) == 2:
            agents_str = f"{formatted_parts[0]} and {formatted_parts[1]}"
        else:
            agents_str = (
                ", ".join(formatted_parts[:-1]) + f", and {formatted_parts[-1]}"
            )

        # Determine verb form based on total count or plurality
        verb = (
            "agree"
            if stance == "agree"
            else ("disagree" if stance == "disagree" else "are unsure")
        )
        verb_singular = (
            "agrees"
            if stance == "agree"
            else ("disagrees" if stance == "disagree" else "is unsure")
        )

        # Choose the appropriate preposition for neutral stances to avoid
        # unnatural phrasing like "are unsure with this statement".
        preposition = "with" if stance in {"agree", "disagree"} else "about"

        # Use singular verb only if we have exactly one agent total
        if len(roles) == 1:
            return f"{agents_str} {verb_singular} {preposition} this statement."
        return f"{agents_str} {verb} {preposition} this statement."

    def _pluralize_role(self, role: str) -> str:
        """Convert a role name to its plural form.

        Args:
            role: singular role name

        Returns:
            plural role name
        """
        # Simple pluralization - add 's' for most cases
        # This handles common cases like "LLM" -> "LLMs", "Expert" -> "Experts"
        if (
            role.endswith("s")
            or role.endswith("x")
            or role.endswith("z")
            or role.endswith("ch")
            or role.endswith("sh")
        ):
            return role + "es"
        if role.endswith("y") and len(role) > 1 and role[-2] not in "aeiou":
            return role[:-1] + "ies"
        return role + "s"

    def _get_role(self, agent_id: int) -> str:
        """Get the role of a specific agent (for future use).

        Args:
            agent_id: ID of the agent to get the role for.

        Returns:
            Role string for the agent.

        Raises:
            KeyError: If agent_id is not found in catalog.
        """
        role = self._agent_roles.get(agent_id)
        if role is None:
            raise KeyError(f"Agent with id={agent_id} not found in catalog")
        if not role:
            raise ValueError(
                f"Agent with id={agent_id} has an empty role in the catalog"
            )
        return role

    def _partition(
        self, neighbor_view: dict[int, float]
    ) -> tuple[list[str], list[str], list[str]]:
        """Partition the neighbor view into agree, disagree, and neutral.

        Args:
            neighbor_view: dictionary mapping neighbor IDs to belief values

        Returns:
            Tuple of (agree_ids, disagree_ids, neutral_ids)
        """
        agree, disagree, neutral = [], [], []

        for k, v in neighbor_view.items():
            if v is not None and v == 1:
                agree.append(str(k))
            elif v is not None and v == 0:
                disagree.append(str(k))
            else:
                neutral.append(str(k))

        # Shuffle each group to reduce order bias
        self._rng.shuffle(agree)
        self._rng.shuffle(disagree)
        self._rng.shuffle(neutral)

        return agree, disagree, neutral

    def as_chat_template(self) -> list[dict[str, str]]:
        """Build the full chat template for querying the LLM.

        Returns:
            chat template as a list of dicts
        """
        assert self._is_ready, (
            "Role, statement, and query names must be set before building chat templates."
        )
        # Get the most recent prompt
        if not self.history:
            raise ValueError("No prompts in history. Call `update` first.")

        latest_round = max(self.history.keys())

        prompt_content = self.history[latest_round].get("prompt", "")
        if prompt_content is None:
            prompt_content = ""

        if self._probe == "zeroshot":
            return [
                {"role": self._system_name, "content": self.context_prompt},
                {"role": self._user_name, "content": prompt_content},
                {"role": self._assistant_name, "content": self.agent_prompt},
            ]

        return [
            {"role": self._system_name, "content": self.context_prompt},
            {"role": self._user_name, "content": prompt_content},
            {"role": self._assistant_name, "content": self.statement},
        ]

    def __str__(self) -> str:
        """Return string representation of Message object.

        Returns:
            String representation with key attributes.
        """
        return (
            f"Message(\n"
            f"  role={self.role},\n"
            f"  statement={self.statement},\n"
            f"  engine={self._engine},\n"
            f"  method={self._method}\n"
            f")"
        )
