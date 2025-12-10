"""
Tests for InferenceScheduler caching behavior
"""

from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from src.InferenceScheduler import LRU, DummyBackend, HFBackend, InferenceScheduler


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
                "model_pool_capacity": 2,
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
            {"seed": 42, "model_pool_capacity": 2, "backend": "dummy"}
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
            {"seed": 42, "model_pool_capacity": 2, "backend": "dummy"}
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
