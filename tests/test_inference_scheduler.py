"""
Tests for InferenceScheduler caching behavior
"""

from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf

from src.core.inference_scheduler import LRU, DummyBackend, HFBackend, InferenceScheduler


class TestLRUCache:
    """Test the LRU cache implementation"""

    def test_lru_basic_get_put(self):
        """Test basic get and put operations"""
        cache = LRU(capacity=2)
        backend1 = DummyBackend("model1", "cpu")
        backend2 = DummyBackend("model2", "cpu")

        cache.put("model1", backend1)
        cache.put("model2", backend2)

        assert cache.get("model1") == backend1
        assert cache.get("model2") == backend2
        assert cache.get("model3") is None

    def test_lru_eviction(self):
        """Test that LRU evicts oldest item when at capacity"""
        cache = LRU(capacity=2)
        backend1 = DummyBackend("model1", "cpu")
        backend2 = DummyBackend("model2", "cpu")
        backend3 = DummyBackend("model3", "cpu")

        cache.put("model1", backend1)
        cache.put("model2", backend2)
        cache.put("model3", backend3)  # Should evict model1

        assert cache.get("model1") is None
        assert cache.get("model2") == backend2
        assert cache.get("model3") == backend3

    def test_lru_move_to_end(self):
        """Test that accessing an item moves it to the end"""
        cache = LRU(capacity=2)
        backend1 = DummyBackend("model1", "cpu")
        backend2 = DummyBackend("model2", "cpu")
        backend3 = DummyBackend("model3", "cpu")

        cache.put("model1", backend1)
        cache.put("model2", backend2)
        cache.get("model1")  # Access model1, making it most recent
        cache.put("model3", backend3)  # Should evict model2, not model1

        assert cache.get("model1") == backend1
        assert cache.get("model2") is None
        assert cache.get("model3") == backend3

    def test_lru_unload_on_eviction(self):
        """Test that evicted backends have unload() called"""
        cache = LRU(capacity=1)

        # Create a mock backend with unload method
        backend1 = Mock()
        backend1.unload = Mock()
        backend2 = Mock()
        backend2.unload = Mock()

        cache.put("model1", backend1)
        cache.put("model2", backend2)  # Should evict model1 and call unload

        backend1.unload.assert_called_once()
        backend2.unload.assert_not_called()


class TestDummyBackend:
    """Test the DummyBackend implementation"""

    def test_dummy_backend_generate(self):
        """Test that DummyBackend generates deterministic output"""
        backend = DummyBackend("test_model", "cpu")

        result1 = backend.generate("test prompt", seed=42)
        result2 = backend.generate("test prompt", seed=42)

        # Same seed and prompt should give same result
        assert result1 == result2
        assert "[test_model]" in result1

    def test_dummy_backend_embed(self):
        """Test that DummyBackend generates embeddings"""
        backend = DummyBackend("test_model", "cpu")

        context = [{"role": "user", "content": "test"}]
        embedding = backend.embed(context)

        # Should return a 16-dimensional vector
        assert len(embedding) == 16
        assert all(0 <= v < 1 for v in embedding)

    def test_dummy_backend_embed_deterministic(self):
        """Test that embeddings are deterministic for same input"""
        backend = DummyBackend("test_model", "cpu")

        context = [{"role": "user", "content": "test"}]
        embedding1 = backend.embed(context)
        embedding2 = backend.embed(context)

        assert embedding1 == embedding2

    def test_dummy_backend_unload(self):
        """Test that unload can be called without errors"""
        backend = DummyBackend("test_model", "cpu")

        # Should not raise any errors
        backend.unload()


class TestHFBackend:
    """Test the HFBackend implementation"""

    def test_hfbackend_init(self):
        """Test HFBackend initialization"""
        backend = HFBackend("test_model", "cpu")

        assert backend.model_name == "test_model"
        assert backend.device == "cpu"
        assert backend._model is None
        assert backend._tokenizer is None
        assert backend._loaded is False

    def test_hfbackend_unload(self):
        """Test that unload properly cleans up"""
        backend = HFBackend("test_model", "cpu")

        # Mock a loaded model
        backend._model = Mock()
        backend._model.to = Mock()
        backend._tokenizer = Mock()
        backend._loaded = True

        backend.unload()

        assert backend._model is None
        assert backend._tokenizer is None
        assert backend._loaded is False


class TestInferenceScheduler:
    """Test the InferenceScheduler implementation"""

    def test_scheduler_init(self):
        """Test InferenceScheduler initialization"""
        cfg = OmegaConf.create(
            {
                "seed": 42,
                "model_pool": {"capacity": 2},
                "device_pool": ["cpu"],
                "backend": "dummy",
            }
        )
        scheduler = InferenceScheduler(cfg)

        assert scheduler.seed == 42
        assert scheduler.pool_capacity == 2
        assert scheduler.devices == ["cpu"]

    def test_scheduler_pick_device_roundrobin(self):
        """Test that pick_device does round-robin selection"""
        cfg = OmegaConf.create({"seed": 42, "device_pool": ["cpu", "cuda:0", "cuda:1"]})
        scheduler = InferenceScheduler(cfg)

        assert scheduler.pick_device() == "cpu"
        assert scheduler.pick_device() == "cuda:0"
        assert scheduler.pick_device() == "cuda:1"
        assert scheduler.pick_device() == "cpu"  # Wraps around

    def test_scheduler_load_model_dummy(self):
        """Test loading a dummy backend"""
        cfg = OmegaConf.create({"seed": 42, "backend": "dummy"})
        scheduler = InferenceScheduler(cfg)

        backend = scheduler.load_model("test_model", "cpu")

        assert isinstance(backend, DummyBackend)
        assert backend.model_name == "test_model"
        assert backend.device == "cpu"

    def test_scheduler_ensure_loaded_caching(self):
        """Test that ensure_loaded caches backends"""
        cfg = OmegaConf.create(
            {"seed": 42, "model_pool": {"capacity": 2}, "backend": "dummy"}
        )
        scheduler = InferenceScheduler(cfg)

        # First access should load the model
        with scheduler.ensure_loaded("model1") as backend1:
            assert isinstance(backend1, DummyBackend)
            backend1_id = id(backend1)

        # Second access should return the same instance
        with scheduler.ensure_loaded("model1") as backend2:
            assert id(backend2) == backend1_id

    def test_scheduler_lru_eviction(self):
        """Test that scheduler evicts models when pool is full"""
        cfg = OmegaConf.create(
            {"seed": 42, "model_pool": {"capacity": 2}, "backend": "dummy"}
        )
        scheduler = InferenceScheduler(cfg)

        # Load three models, capacity is 2
        with scheduler.ensure_loaded("model1"):
            pass
        with scheduler.ensure_loaded("model2"):
            pass

        # Both should be in cache
        assert scheduler.model_pool.get("model1") is not None
        assert scheduler.model_pool.get("model2") is not None

        # Load model3, should evict model1 (least recently used)
        with scheduler.ensure_loaded("model3"):
            pass

        # model1 should be evicted (was least recently used)
        assert scheduler.model_pool.get("model1") is None
        # model2 and model3 should still be in cache
        assert scheduler.model_pool.get("model2") is not None
        assert scheduler.model_pool.get("model3") is not None

    def test_scheduler_skips_device_acquisition_for_same_model(self):
        """Test that acquire_device is not called when the same model is reused."""
        cfg = OmegaConf.create(
            {"seed": 42, "model_pool": {"capacity": 2}, "backend": "dummy"}
        )
        scheduler = InferenceScheduler(cfg)

        # Create mock backends that track acquire_device calls
        mock_backend = Mock()
        mock_backend.acquire_device = Mock()
        mock_backend.release_device = Mock()

        # Replace load_model to return our mock
        scheduler.load_model = Mock(return_value=mock_backend)

        # First access should call acquire_device
        with scheduler.ensure_loaded("model1"):
            pass
        assert mock_backend.acquire_device.call_count == 1
        assert mock_backend.release_device.call_count == 0  # Not released anymore

        # Second access to same model should NOT call acquire_device again
        with scheduler.ensure_loaded("model1"):
            pass
        assert mock_backend.acquire_device.call_count == 1  # Still 1, not called again
        assert mock_backend.release_device.call_count == 0

    def test_scheduler_acquires_device_when_switching_models(self):
        """Test that acquire_device is called when switching to a different model."""
        cfg = OmegaConf.create(
            {"seed": 42, "model_pool": {"capacity": 2}, "backend": "dummy"}
        )
        scheduler = InferenceScheduler(cfg)

        # Create two mock backends
        mock_backend1 = Mock()
        mock_backend1.acquire_device = Mock()
        mock_backend1.release_device = Mock()

        mock_backend2 = Mock()
        mock_backend2.acquire_device = Mock()
        mock_backend2.release_device = Mock()

        # Replace load_model to return different mocks based on model name
        def mock_load_model(model_name, device):
            if model_name == "model1":
                return mock_backend1
            return mock_backend2

        scheduler.load_model = mock_load_model

        # First access should call acquire_device for model1
        with scheduler.ensure_loaded("model1"):
            pass
        assert mock_backend1.acquire_device.call_count == 1
        assert mock_backend1.release_device.call_count == 0

        # Switch to model2 should release model1 and acquire model2
        with scheduler.ensure_loaded("model2"):
            pass
        assert mock_backend1.release_device.call_count == 1  # Released when switching
        assert mock_backend2.acquire_device.call_count == 1

    def test_scheduler_active_model_tracking(self):
        """Test that _active_model is correctly tracked."""
        cfg = OmegaConf.create(
            {"seed": 42, "model_pool": {"capacity": 2}, "backend": "dummy"}
        )
        scheduler = InferenceScheduler(cfg)

        # Initially no active model
        assert scheduler._active_model is None

        # After loading model1, it should be active
        with scheduler.ensure_loaded("model1"):
            pass
        assert scheduler._active_model == "model1"

        # After loading model2, model2 should be active
        with scheduler.ensure_loaded("model2"):
            pass
        assert scheduler._active_model == "model2"

        # Reloading model1 should switch back
        with scheduler.ensure_loaded("model1"):
            pass
        assert scheduler._active_model == "model1"

    def test_scheduler_clears_active_model_on_eviction(self):
        """Test that _active_model is cleared when the active model is evicted."""
        cfg = OmegaConf.create(
            {
                "seed": 42,
                "model_pool": {"capacity": 1},  # Only 1 model can be cached
                "backend": "dummy",
            }
        )
        scheduler = InferenceScheduler(cfg)

        # Load model1, it becomes active
        with scheduler.ensure_loaded("model1"):
            pass
        assert scheduler._active_model == "model1"

        # Load model2, which should evict model1 and clear _active_model
        with scheduler.ensure_loaded("model2"):
            pass
        # model2 should now be active (model1 was evicted)
        assert scheduler._active_model == "model2"
        # model1 should be evicted from cache
        assert scheduler.model_pool.get("model1") is None

    def test_lru_eviction_callback(self):
        """Test that LRU eviction callback is called when a model is evicted."""
        evicted_keys = []

        def on_evict(key):
            evicted_keys.append(key)

        cache = LRU(capacity=1, on_evict=on_evict)
        backend1 = DummyBackend("model1", "cpu")
        backend2 = DummyBackend("model2", "cpu")

        cache.put("model1", backend1)
        assert len(evicted_keys) == 0

        cache.put("model2", backend2)  # Should evict model1
        assert evicted_keys == ["model1"]

    def test_prompt_logits_cache_reuses_result(self):
        """Scheduler caches logits for identical prompts per agent+model."""
        cfg = OmegaConf.create(
            {
                "seed": 123,
                "model_pool": {"capacity": 1},
                "backend": "dummy",
                "prompt_cache_capacity": 8,
            }
        )
        scheduler = InferenceScheduler(cfg)

        # Stub backend to count invocations
        mock_backend = Mock()
        mock_backend.acquire_device = Mock()
        mock_backend.release_device = Mock()
        # deterministic logits
        logits = torch.randn(1, 100)
        mock_backend.get_logits = Mock(return_value=logits)
        scheduler.load_model = Mock(return_value=mock_backend)

        class StubAgent:
            id = 1
            model_name = "modelA"

            def prepare_round(self, t, neighbor_view):
                pass

            @property
            def current_message(self):
                return [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Is the statement true?"},
                ]

        agent = StubAgent()

        out1 = scheduler.get_logits(agent, t=0)
        out2 = scheduler.get_logits(agent, t=1)  # same prompt

        # Backend should have been called only once due to caching
        assert mock_backend.get_logits.call_count == 1
        # Outputs should be identical tensors
        assert torch.equal(out1, out2)
        # Cache should contain exactly one entry
        assert len(scheduler._logits_cache) == 1

    def test_prompt_cache_shares_across_agents_with_same_role(self):
        """Identical prompts for same role+model share cached logits."""
        cfg = OmegaConf.create(
            {
                "seed": 123,
                "model_pool": {"capacity": 1},
                "backend": "dummy",
                "prompt_cache_capacity": 8,
            }
        )
        scheduler = InferenceScheduler(cfg)

        mock_backend = Mock()
        mock_backend.acquire_device = Mock()
        mock_backend.release_device = Mock()
        logits = torch.randn(1, 100)
        mock_backend.get_logits = Mock(return_value=logits)
        scheduler.load_model = Mock(return_value=mock_backend)

        class StubAgent1:
            id = 1
            role = "LLM"
            model_name = "modelA"

            def prepare_round(self, t, neighbor_view):
                pass

            @property
            def current_message(self):
                return [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Is the statement true?"},
                ]

        class StubAgent2:
            id = 2
            role = "LLM"
            model_name = "modelA"

            def prepare_round(self, t, neighbor_view):
                pass

            @property
            def current_message(self):
                return [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Is the statement true?"},
                ]

        a1 = StubAgent1()
        a2 = StubAgent2()

        _ = scheduler.get_logits(a1, t=0)
        _ = scheduler.get_logits(a2, t=0)

        # Backend should be called once (same role+model+prompt -> shared key)
        assert mock_backend.get_logits.call_count == 1
        # Cache should contain one entry
        assert len(scheduler._logits_cache) == 1

    def test_prompt_cache_miss_on_role_change(self):
        """Same prompt and model but different role should miss cache."""
        cfg = OmegaConf.create(
            {
                "seed": 123,
                "model_pool": {"capacity": 1},
                "backend": "dummy",
                "prompt_cache_capacity": 8,
            }
        )
        scheduler = InferenceScheduler(cfg)

        mock_backend = Mock()
        mock_backend.acquire_device = Mock()
        mock_backend.release_device = Mock()
        logits = torch.randn(1, 100)
        mock_backend.get_logits = Mock(return_value=logits)
        scheduler.load_model = Mock(return_value=mock_backend)

        class AgentLLM:
            role = "LLM"
            model_name = "modelA"

            def prepare_round(self, t, neighbor_view):
                pass

            @property
            def current_message(self):
                return [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Is the statement true?"},
                ]

        class AgentParticipant:
            role = "Participant"
            model_name = "modelA"

            def prepare_round(self, t, neighbor_view):
                pass

            @property
            def current_message(self):
                return [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Is the statement true?"},
                ]

        a1 = AgentLLM()
        a2 = AgentParticipant()

        _ = scheduler.get_logits(a1, t=0)
        _ = scheduler.get_logits(a2, t=0)

        # Backend should be called twice (different role -> different key)
        assert mock_backend.get_logits.call_count == 2
        assert len(scheduler._logits_cache) == 2

    def test_prompt_cache_miss_on_model_change(self):
        """Same prompt and role but different model should miss cache."""
        cfg = OmegaConf.create(
            {
                "seed": 123,
                "model_pool": {"capacity": 1},
                "backend": "dummy",
                "prompt_cache_capacity": 8,
            }
        )
        scheduler = InferenceScheduler(cfg)

        mock_backend = Mock()
        mock_backend.acquire_device = Mock()
        mock_backend.release_device = Mock()
        logits = torch.randn(1, 100)
        mock_backend.get_logits = Mock(return_value=logits)
        # Return same mock for simplicity; keying is what matters
        scheduler.load_model = Mock(return_value=mock_backend)

        class AgentA:
            role = "LLM"
            model_name = "modelA"

            def prepare_round(self, t, neighbor_view):
                pass

            @property
            def current_message(self):
                return [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Is the statement true?"},
                ]

        class AgentB:
            role = "LLM"
            model_name = "modelB"

            def prepare_round(self, t, neighbor_view):
                pass

            @property
            def current_message(self):
                return [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Is the statement true?"},
                ]

        a = AgentA()
        b = AgentB()

        _ = scheduler.get_logits(a, t=0)
        _ = scheduler.get_logits(b, t=0)

        # Different model -> different key -> two calls
        assert mock_backend.get_logits.call_count == 2
        assert len(scheduler._logits_cache) == 2

    def test_prompt_cache_miss_on_content_change(self):
        """Minor content changes should produce a cache miss."""
        cfg = OmegaConf.create(
            {
                "seed": 123,
                "model_pool": {"capacity": 1},
                "backend": "dummy",
                "prompt_cache_capacity": 8,
            }
        )
        scheduler = InferenceScheduler(cfg)

        mock_backend = Mock()
        mock_backend.acquire_device = Mock()
        mock_backend.release_device = Mock()
        logits = torch.randn(1, 100)
        mock_backend.get_logits = Mock(return_value=logits)
        scheduler.load_model = Mock(return_value=mock_backend)

        class Agent:
            role = "LLM"
            model_name = "modelA"

            def __init__(self, suffix=""):
                self.suffix = suffix

            def prepare_round(self, t, neighbor_view):
                pass

            @property
            def current_message(self):
                return [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Is the statement true?" + self.suffix},
                ]

        a1 = Agent("")
        a2 = Agent(" ")  # trailing space changes content

        _ = scheduler.get_logits(a1, t=0)
        _ = scheduler.get_logits(a2, t=0)

        # Content change -> different key -> two calls
        assert mock_backend.get_logits.call_count == 2
        assert len(scheduler._logits_cache) == 2

    def test_prompt_cache_eviction_on_capacity(self):
        """LRU eviction should remove old entry and cause a miss later."""
        cfg = OmegaConf.create(
            {
                "seed": 123,
                "model_pool": {"capacity": 1},
                "backend": "dummy",
                "prompt_cache": {"capacity": 1, "unbounded": False},
            }
        )
        scheduler = InferenceScheduler(cfg)

        mock_backend = Mock()
        mock_backend.acquire_device = Mock()
        mock_backend.release_device = Mock()
        logits = torch.randn(1, 100)
        mock_backend.get_logits = Mock(return_value=logits)
        scheduler.load_model = Mock(return_value=mock_backend)

        class Agent:
            role = "LLM"
            model_name = "modelA"

            def __init__(self, msg):
                self.msg = msg

            def prepare_round(self, t, neighbor_view):
                pass

            @property
            def current_message(self):
                return [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": self.msg},
                ]

        a = Agent("Is the statement true?")
        b = Agent("Is the statement false?")

        _ = scheduler.get_logits(a, t=0)  # fills cache with A
        _ = scheduler.get_logits(b, t=0)  # evicts A, keeps B

        # Now querying A again should be a miss (since it was evicted)
        _ = scheduler.get_logits(a, t=1)

        assert mock_backend.get_logits.call_count == 3
        assert len(scheduler._logits_cache) == 1

    def test_prompt_cache_unbounded_no_eviction(self):
        """With unbounded cache, old entries are retained and reused."""
        cfg = OmegaConf.create(
            {
                "seed": 123,
                "model_pool": {"capacity": 1},
                "backend": "dummy",
                "prompt_cache_capacity": 1,  # capacity ignored when unbounded
                "prompt_cache_unbounded": True,
            }
        )
        scheduler = InferenceScheduler(cfg)

        mock_backend = Mock()
        mock_backend.acquire_device = Mock()
        mock_backend.release_device = Mock()
        logits = torch.randn(1, 100)
        mock_backend.get_logits = Mock(return_value=logits)
        scheduler.load_model = Mock(return_value=mock_backend)

        class Agent:
            role = "LLM"
            model_name = "modelA"

            def __init__(self, msg):
                self.msg = msg

            def prepare_round(self, t, neighbor_view):
                pass

            @property
            def current_message(self):
                return [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": self.msg},
                ]

        a = Agent("Is the statement true?")
        b = Agent("Is the statement false?")
        c = Agent("Is the statement uncertain?")

        _ = scheduler.get_logits(a, t=0)  # cache A
        _ = scheduler.get_logits(b, t=0)  # cache B
        _ = scheduler.get_logits(c, t=0)  # cache C

        # Unbounded: cache should retain all entries
        assert len(scheduler._logits_cache) == 3

        # Re-query A should hit cache (no new backend call)
        _ = scheduler.get_logits(a, t=1)
        assert mock_backend.get_logits.call_count == 3

    def test_prompt_cache_miss_on_message_order_change(self):
        """Changing message order should miss cache even if content is same."""
        cfg = OmegaConf.create(
            {
                "seed": 123,
                "model_pool": {"capacity": 1},
                "backend": "dummy",
                "prompt_cache_capacity": 8,
            }
        )
        scheduler = InferenceScheduler(cfg)

        mock_backend = Mock()
        mock_backend.acquire_device = Mock()
        mock_backend.release_device = Mock()
        logits = torch.randn(1, 100)
        mock_backend.get_logits = Mock(return_value=logits)
        scheduler.load_model = Mock(return_value=mock_backend)

        class Agent1:
            role = "LLM"
            model_name = "modelA"

            def prepare_round(self, t, neighbor_view):
                pass

            @property
            def current_message(self):
                return [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Is the statement true?"},
                ]

        class Agent2:
            role = "LLM"
            model_name = "modelA"

            def prepare_round(self, t, neighbor_view):
                pass

            @property
            def current_message(self):
                return [
                    {"role": "user", "content": "Is the statement true?"},
                    {"role": "system", "content": "You are helpful."},
                ]

        _ = scheduler.get_logits(Agent1(), t=0)
        _ = scheduler.get_logits(Agent2(), t=0)

        # Order change -> different key -> two calls
        assert mock_backend.get_logits.call_count == 2
        assert len(scheduler._logits_cache) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
