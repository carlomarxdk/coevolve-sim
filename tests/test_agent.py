"""
Tests for Agent classes
"""

from unittest.mock import Mock

import pytest

from src.Agent import BaseAgent, ExpertAgent, LLMAgent
from src.Message import Message
from src.Probe import Probe


class TestBaseAgent:
    """Test the BaseAgent class"""

    @pytest.fixture
    def mock_message(self):
        """Create a mock Message object"""
        msg = Mock(spec=Message)
        msg.set_role_and_context = Mock()
        msg.set_query_names = Mock()
        msg.set_statement = Mock()
        msg.update = Mock()
        msg.as_chat_template = Mock(return_value=[{"role": "user", "content": "test"}])
        return msg

    @pytest.fixture
    def mock_probe(self):
        """Create a mock Probe object"""
        probe = Mock(spec=Probe)
        probe.load_probe = Mock()
        return probe

    @pytest.fixture
    def basic_config(self):
        """Create a basic config for agent"""
        return {
            "model": {
                "query": {"user": "user", "assistant": "assistant", "system": "system"}
            }
        }

    def test_base_agent_initialization(self, mock_message, mock_probe, basic_config):
        """Test BaseAgent initialization"""
        agent = BaseAgent(
            id=1,
            model_name="test_model",
            cfg=basic_config,
            message=mock_message,
            probe=mock_probe,
            role="Moderator",
        )

        assert agent.id == 1
        assert agent.model_name == "test_model"
        assert agent.role == "Moderator"
        assert agent.message == mock_message
        assert agent.probe == mock_probe
        assert isinstance(agent.beliefs, dict)

        # Check that post_init was called
        mock_message.set_role_and_context.assert_called_once_with(role="Moderator")
        mock_message.set_query_names.assert_called_once_with(
            user_name="user", assistant_name="assistant", system_name="system"
        )
        mock_probe.load_probe.assert_called_once()

    def test_set_statement(self, mock_message, mock_probe, basic_config):
        """Test set_statement method"""
        agent = BaseAgent(
            id=1,
            model_name="test_model",
            cfg=basic_config,
            message=mock_message,
            probe=mock_probe,
        )

        agent.set_statement("The Earth is round.")
        mock_message.set_statement.assert_called_once_with("The Earth is round.")

    def test_prepare_round(self, mock_message, mock_probe, basic_config):
        """Test prepare_round method"""
        agent = BaseAgent(
            id=1,
            model_name="test_model",
            cfg=basic_config,
            message=mock_message,
            probe=mock_probe,
        )

        neighbor_view = {2: 0.8, 3: 0.6}
        agent.prepare_round(t=1, neighbor_view=neighbor_view)

        mock_message.update.assert_called_once_with(1, neighbor_view)

    def test_current_message_property(self, mock_message, mock_probe, basic_config):
        """Test current_message property"""
        agent = BaseAgent(
            id=1,
            model_name="test_model",
            cfg=basic_config,
            message=mock_message,
            probe=mock_probe,
        )

        result = agent.current_message
        assert result == [{"role": "user", "content": "test"}]
        mock_message.as_chat_template.assert_called_once()

    def test_set_belief(self, mock_message, mock_probe, basic_config):
        """Test set_belief method"""
        agent = BaseAgent(
            id=1,
            model_name="test_model",
            cfg=basic_config,
            message=mock_message,
            probe=mock_probe,
        )

        agent.set_belief(0.75, t=1)
        assert agent.beliefs[1] == 0.75

        agent.set_belief(0.85, t=2)
        assert agent.beliefs[2] == 0.85

    def test_current_belief(self, mock_message, mock_probe, basic_config):
        """Test current_belief method"""
        agent = BaseAgent(
            id=1,
            model_name="test_model",
            cfg=basic_config,
            message=mock_message,
            probe=mock_probe,
        )

        agent.set_belief(0.75, t=1)

        assert agent.current_belief(1) == 0.75
        assert agent.current_belief(2) is None  # Non-existent round

    def test_beliefs_tracking_over_time(self, mock_message, mock_probe, basic_config):
        """Test that beliefs are tracked correctly over multiple rounds"""
        agent = BaseAgent(
            id=1,
            model_name="test_model",
            cfg=basic_config,
            message=mock_message,
            probe=mock_probe,
        )

        agent.set_belief(0.5, t=0)
        agent.set_belief(0.6, t=1)
        agent.set_belief(0.7, t=2)

        assert agent.current_belief(0) == 0.5
        assert agent.current_belief(1) == 0.6
        assert agent.current_belief(2) == 0.7


class TestLLMAgent:
    """Test the LLMAgent class"""

    @pytest.fixture
    def mock_message(self):
        """Create a mock Message object"""
        msg = Mock(spec=Message)
        msg.set_role_and_context = Mock()
        msg.set_query_names = Mock()
        msg.as_chat_template = Mock(return_value=[{"role": "user", "content": "test"}])
        return msg

    @pytest.fixture
    def mock_probe(self):
        """Create a mock Probe object"""
        probe = Mock(spec=Probe)
        probe.load_probe = Mock()
        return probe

    @pytest.fixture
    def basic_config(self):
        """Create a basic config for agent"""
        return {
            "model": {
                "query": {"user": "user", "assistant": "assistant", "system": "system"}
            }
        }

    def test_llm_agent_initialization(self, mock_message, mock_probe, basic_config):
        """Test LLMAgent initialization"""
        agent = LLMAgent(
            id=1,
            model_name="llama-base",
            cfg=basic_config,
            message=mock_message,
            probe=mock_probe,
            role="Scientist",
        )

        assert isinstance(agent, BaseAgent)
        assert isinstance(agent, LLMAgent)
        assert agent.model_name == "llama-base"
        assert agent.role == "Scientist"

    def test_llm_agent_generate(self, mock_message, mock_probe, basic_config):
        """Test LLMAgent generate method"""
        agent = LLMAgent(
            id=1,
            model_name="llama-base",
            cfg=basic_config,
            message=mock_message,
            probe=mock_probe,
        )

        mock_scheduler = Mock()
        mock_scheduler.query_llm = Mock(return_value="Generated response")

        result = agent.generate("Test prompt", mock_scheduler)

        assert result == "Generated response"
        mock_scheduler.query_llm.assert_called_once_with(agent, "Test prompt")


class TestExpertAgent:
    """Test the ExpertAgent class"""

    @pytest.fixture
    def mock_message(self):
        """Create a mock Message object"""
        msg = Mock(spec=Message)
        msg.set_role_and_context = Mock()
        msg.set_query_names = Mock()
        msg.as_chat_template = Mock(return_value=[{"role": "user", "content": "test"}])
        return msg

    @pytest.fixture
    def mock_probe(self):
        """Create a mock Probe object"""
        probe = Mock(spec=Probe)
        probe.load_probe = Mock()
        return probe

    @pytest.fixture
    def basic_config(self):
        """Create a basic config for agent"""
        return {
            "model": {
                "query": {"user": "user", "assistant": "assistant", "system": "system"}
            }
        }

    def test_expert_agent_initialization(self, mock_message, mock_probe, basic_config):
        """Test ExpertAgent initialization"""
        agent = ExpertAgent(
            id=1,
            model_name="expert",
            cfg=basic_config,
            message=mock_message,
            probe=mock_probe,
            role="Expert",
        )

        assert isinstance(agent, BaseAgent)
        assert isinstance(agent, ExpertAgent)
        assert agent.model_name == "expert"
        assert agent.role == "Expert"

    def test_expert_agent_generate(self, mock_message, mock_probe, basic_config):
        """Test ExpertAgent generate method"""
        agent = ExpertAgent(
            id=1,
            model_name="expert",
            cfg=basic_config,
            message=mock_message,
            probe=mock_probe,
        )

        mock_scheduler = Mock()
        mock_scheduler.query_llm = Mock(return_value="Expert response")

        result = agent.generate("Test prompt", mock_scheduler)

        assert result == "Expert response"
        mock_scheduler.query_llm.assert_called_once_with(agent, "Test prompt")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
