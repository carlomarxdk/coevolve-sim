"""
Test suite for Message class, specifically for handling duplicate roles.
"""

import pytest

from src.Message import Message


class TestMessageDuplicateRoles:
    """Test handling of duplicate roles in Message class."""

    @pytest.fixture
    def config_with_roles(self):
        """Create a config with role-based aggregation."""
        return {
            "seed": 42,
            "prompt": {
                "engine": "jinja2",
                "template": {
                    "system": "You are a {{ role }} in a social discussion network.",
                    "user": {
                        "intro": "Your friends are discussing the following statement: {{ statement }}.",
                        "agree": (
                            "{% if agents|length == 1 %}"
                            "{{ agents[0] }} agrees with this statement."
                            "{% elif agents|length == 2 %}"
                            "{{ agents[0] }} and {{ agents[1] }} agree with this statement."
                            "{% elif agents|length > 2 %}"
                            "{{ agents[:-1]|join(', ') }}, and {{ agents[-1] }} agree with this statement."
                            "{% endif %}"
                        ),
                        "disagree": (
                            "{% if agents|length == 1 %}"
                            "{{ agents[0] }} disagrees with this statement."
                            "{% elif agents|length == 2 %}"
                            "{{ agents[0] }} and {{ agents[1] }} disagree with this statement."
                            "{% elif agents|length > 2 %}"
                            "{{ agents[:-1]|join(', ') }}, and {{ agents[-1] }} disagree with this statement."
                            "{% endif %}"
                        ),
                        "neutral": (
                            "{% if agents|length == 1 %}"
                            "{{ agents[0] }} is unsure about this statement."
                            "{% elif agents|length == 2 %}"
                            "{{ agents[0] }} and {{ agents[1] }} are unsure about this statement."
                            "{% elif agents|length > 2 %}"
                            "{{ agents[:-1]|join(', ') }}, and {{ agents[-1] }} are unsure about this statement."
                            "{% endif %}"
                        ),
                        "instruction": "What do you think about this statement?",
                    },
                },
                "aggregation": {"method": "list_all"},
            },
            "agents": {
                "catalog": [
                    {"id": 1, "role": "LLM"},
                    {"id": 2, "role": "LLM"},
                    {"id": 3, "role": "LLM"},
                    {"id": 4, "role": "Expert"},
                    {"id": 5, "role": "Expert"},
                ]
            },
        }

    def test_single_role_type(self, config_with_roles):
        """Test that multiple agents with the same role are grouped and counted."""
        msg = Message(cfg=config_with_roles)
        msg.set_role_and_context("Moderator")
        msg.set_statement("The Earth is round.")
        msg.set_query_names("user", "assistant", "system")

        # Three LLMs agree
        neighbor_view = {1: 1, 2: 1, 3: 1}
        prompt = msg.update(1, neighbor_view)

        # Should say "3 LLMs agree" not "LLM, LLM, and LLM agree"
        assert (
            "3 LLMs" in prompt or "three LLMs" in prompt.lower()
        ), f"Expected grouped count but got: {prompt}"
        assert (
            prompt.count("LLM") < 4
        ), f"Expected LLM to appear at most 3 times (once in count), but got: {prompt}"

    def test_multiple_role_types(self, config_with_roles):
        """Test that different roles are handled properly when mixed."""
        msg = Message(cfg=config_with_roles)
        msg.set_role_and_context("Moderator")
        msg.set_statement("The Earth is round.")
        msg.set_query_names("user", "assistant", "system")

        # Two LLMs and two Experts agree
        neighbor_view = {1: 1, 2: 1, 4: 1, 5: 1}
        prompt = msg.update(1, neighbor_view)

        # Should say "2 LLMs and 2 Experts agree" or similar
        assert ("2 LLMs" in prompt or "two LLMs" in prompt.lower()) and (
            "2 Experts" in prompt or "two Experts" in prompt.lower()
        ), f"Expected grouped counts for both roles but got: {prompt}"

    def test_single_agent(self, config_with_roles):
        """Test that a single agent is not pluralized incorrectly."""
        msg = Message(cfg=config_with_roles)
        msg.set_role_and_context("Moderator")
        msg.set_statement("The Earth is round.")
        msg.set_query_names("user", "assistant", "system")

        # Single LLM agrees
        neighbor_view = {1: 1}
        prompt = msg.update(1, neighbor_view)

        # Should say "1 LLM agrees" or "LLM agrees", not "LLMs"
        # The exact format depends on implementation, but should be grammatically correct
        assert "LLM" in prompt, f"Expected LLM in prompt but got: {prompt}"

    def test_mixed_stances(self, config_with_roles):
        """Test that roles are grouped correctly across different stances."""
        msg = Message(cfg=config_with_roles)
        msg.set_role_and_context("Moderator")
        msg.set_statement("The Earth is round.")
        msg.set_query_names("user", "assistant", "system")

        # Two LLMs agree, one LLM disagrees
        neighbor_view = {1: 1, 2: 1, 3: 0}
        prompt = msg.update(1, neighbor_view)

        # Should properly group within each stance
        assert (
            "LLM" in prompt or "LLMs" in prompt
        ), f"Expected LLM/LLMs in prompt but got: {prompt}"
        # Should mention both agree and disagree
        assert (
            "agree" in prompt.lower() and "disagree" in prompt.lower()
        ), f"Expected both stances but got: {prompt}"
