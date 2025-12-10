"""
Tests for ZeroShotProbe class
"""

from unittest.mock import Mock

import numpy as np
import pytest

from src.Probe import ZeroShotProbe


class TestZeroShotProbe:
    """Test the ZeroShotProbe class"""

    @pytest.fixture
    def basic_config(self):
        """Create a basic probe config"""
        return {"name": "zeroshot"}

    @pytest.fixture
    def model_config(self):
        """Create a basic model config"""
        return {"model": {"name": "llama-base"}}

    @pytest.fixture
    def mock_io(self):
        """Create a mock IOManager"""
        return Mock()

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer"""
        tokenizer = Mock()

        # Mock encode method to return token IDs
        # Token '1' -> [16], ' 1' -> [352]
        # Token '2' -> [17], ' 2' -> [353]
        # etc.
        def encode_fn(text, add_special_tokens=False):
            encoding_map = {
                "1": [16],
                " 1": [352],
                "2": [17],
                " 2": [353],
                "3": [18],
                " 3": [354],
                "4": [19],
                " 4": [355],
                "5": [20],
                " 5": [356],
                "6": [21],
                " 6": [357],
            }
            return encoding_map.get(text, [0])

        tokenizer.encode = Mock(side_effect=encode_fn)
        return tokenizer

    def test_zeroshot_probe_initialization(self, basic_config, model_config, mock_io):
        """Test ZeroShotProbe initialization"""
        probe = ZeroShotProbe(cfg=basic_config, model_cfg=model_config, io=mock_io)

        assert probe.cfg == basic_config
        assert probe.model_cfg == model_config
        assert probe.io == mock_io
        assert probe._name == "zeroshot"
        assert probe.tokenizer is None
        assert probe.token_ids == {}

    def test_set_tokenizer(self, basic_config, model_config, mock_io, mock_tokenizer):
        """Test setting tokenizer and preparing token IDs"""
        probe = ZeroShotProbe(cfg=basic_config, model_cfg=model_config, io=mock_io)
        probe.set_tokenizer(mock_tokenizer)

        assert probe.tokenizer == mock_tokenizer
        assert len(probe.token_ids) == 6  # Should have token IDs for 1-6

        # Check that each option has token IDs
        for option in ["1", "2", "3", "4", "5", "6"]:
            assert option in probe.token_ids
            assert len(probe.token_ids[option]) > 0

    def test_score_with_true_prediction(
        self, basic_config, model_config, mock_io, mock_tokenizer
    ):
        """Test scoring when model predicts True (option 1)"""
        probe = ZeroShotProbe(cfg=basic_config, model_cfg=model_config, io=mock_io)
        probe.set_tokenizer(mock_tokenizer)

        # Create logits favoring token 16 (option '1')
        vocab_size = 500
        logits = np.zeros((1, vocab_size), dtype=np.float32)
        logits[0, 16] = 10.0  # High logit for token '1'
        logits[0, 352] = 9.0  # High logit for ' 1'

        predicted_class, confidence = probe.score(logits)

        assert predicted_class == 1.0  # Should predict True
        assert confidence > 0.5  # Should have high confidence

    def test_score_with_false_prediction(
        self, basic_config, model_config, mock_io, mock_tokenizer
    ):
        """Test scoring when model predicts False (option 2)"""
        probe = ZeroShotProbe(cfg=basic_config, model_cfg=model_config, io=mock_io)
        probe.set_tokenizer(mock_tokenizer)

        # Create logits favoring token 17 (option '2')
        vocab_size = 500
        logits = np.zeros((1, vocab_size), dtype=np.float32)
        logits[0, 17] = 10.0  # High logit for token '2'
        logits[0, 353] = 9.0  # High logit for ' 2'

        predicted_class, confidence = probe.score(logits)

        assert predicted_class == 0.0  # Should predict False
        assert confidence > 0.5  # Should have high confidence

    def test_score_with_uncertain_prediction(
        self, basic_config, model_config, mock_io, mock_tokenizer
    ):
        """Test scoring when model predicts Uncertain (options 3-6)"""
        probe = ZeroShotProbe(cfg=basic_config, model_cfg=model_config, io=mock_io)
        probe.set_tokenizer(mock_tokenizer)

        # Create logits favoring tokens 18-21 (options '3'-'6')
        vocab_size = 500
        logits = np.zeros((1, vocab_size), dtype=np.float32)
        for i in [18, 19, 20, 21]:  # Tokens for '3', '4', '5', '6'
            logits[0, i] = 8.0

        predicted_class, confidence = probe.score(logits)

        assert predicted_class == -1.0  # Should predict Uncertain
        assert confidence > 0.5  # Should have high confidence

    def test_score_without_tokenizer_raises_error(
        self, basic_config, model_config, mock_io
    ):
        """Test that scoring without tokenizer raises error"""
        probe = ZeroShotProbe(cfg=basic_config, model_cfg=model_config, io=mock_io)

        logits = np.zeros((1, 500), dtype=np.float32)

        with pytest.raises(RuntimeError, match="Tokenizer not set"):
            probe.score(logits)

    def test_score_handles_1d_logits(
        self, basic_config, model_config, mock_io, mock_tokenizer
    ):
        """Test that score method handles 1D logits input"""
        probe = ZeroShotProbe(cfg=basic_config, model_cfg=model_config, io=mock_io)
        probe.set_tokenizer(mock_tokenizer)

        # Create 1D logits (will be reshaped internally)
        vocab_size = 500
        logits = np.zeros(vocab_size, dtype=np.float32)
        logits[16] = 10.0  # Favor option '1'

        predicted_class, confidence = probe.score(logits)

        assert predicted_class == 1.0
        assert confidence > 0.0

    def test_get_unique_token_ids_filters_shared(
        self, basic_config, model_config, mock_io
    ):
        """Test that shared tokens are filtered out"""
        probe = ZeroShotProbe(cfg=basic_config, model_cfg=model_config, io=mock_io)

        # Mock tokenizer that returns overlapping tokens
        tokenizer = Mock()

        def encode_fn(text, add_special_tokens=False):
            if text == "1":
                return [10, 99]  # 99 is shared
            elif text == " 1":
                return [11]
            elif text == "2":
                return [20, 99]  # 99 is shared
            elif text == " 2":
                return [21]
            else:
                return [0]

        tokenizer.encode = Mock(side_effect=encode_fn)
        probe.tokenizer = tokenizer  # Set tokenizer before calling the method

        token_dict = {"1": ["1", " 1"], "2": ["2", " 2"]}

        result = probe._get_unique_token_ids(token_dict)

        # Token 99 should be filtered out as it appears in both
        assert 99 not in result["1"]
        assert 99 not in result["2"]
        assert 10 in result["1"]
        assert 20 in result["2"]

    def test_probabilities_sum_to_one(
        self, basic_config, model_config, mock_io, mock_tokenizer
    ):
        """Test that collected probabilities always sum to 1"""
        probe = ZeroShotProbe(cfg=basic_config, model_cfg=model_config, io=mock_io)
        probe.set_tokenizer(mock_tokenizer)

        # Test with various logit distributions
        vocab_size = 500

        # Test case 1: High confidence on option 1
        logits1 = np.zeros((1, vocab_size), dtype=np.float32)
        logits1[0, 16] = 10.0  # High logit for token '1'
        probs1 = probe._collect_probabilities(logits1)
        assert (
            abs(sum(probs1.values()) - 1.0) < 1e-6
        ), f"Probabilities don't sum to 1: {sum(probs1.values())}"

        # Test case 2: Uniform distribution
        logits2 = np.ones((1, vocab_size), dtype=np.float32)
        probs2 = probe._collect_probabilities(logits2)
        assert (
            abs(sum(probs2.values()) - 1.0) < 1e-6
        ), f"Probabilities don't sum to 1: {sum(probs2.values())}"

        # Test case 3: Random distribution
        logits3 = np.random.randn(1, vocab_size).astype(np.float32)
        probs3 = probe._collect_probabilities(logits3)
        assert (
            abs(sum(probs3.values()) - 1.0) < 1e-6
        ), f"Probabilities don't sum to 1: {sum(probs3.values())}"

        # Test case 4: Multiple high-probability options
        logits4 = np.zeros((1, vocab_size), dtype=np.float32)
        for i in [16, 17, 18]:  # Options 1, 2, 3
            logits4[0, i] = 5.0
        probs4 = probe._collect_probabilities(logits4)
        assert (
            abs(sum(probs4.values()) - 1.0) < 1e-6
        ), f"Probabilities don't sum to 1: {sum(probs4.values())}"

        # Verify "else" category exists in all cases
        assert "else" in probs1
        assert "else" in probs2
        assert "else" in probs3
        assert "else" in probs4


class TestZeroShotProbeIntegration:
    """Integration tests for ZeroShotProbe"""

    def test_full_pipeline(self):
        """Test complete scoring pipeline"""
        cfg = {"name": "zeroshot"}
        model_cfg = {"model": {"name": "llama-base"}}
        io = Mock()

        probe = ZeroShotProbe(cfg=cfg, model_cfg=model_cfg, io=io)

        # Create simple mock tokenizer
        def simple_encode(text, add_special_tokens=False):
            """Simple encoding based on last character."""
            return [ord(text[-1])] if text else [0]

        tokenizer = Mock()
        tokenizer.encode = Mock(side_effect=simple_encode)

        probe.set_tokenizer(tokenizer)

        # Create logits
        vocab_size = 500
        logits = np.random.randn(1, vocab_size).astype(np.float32)

        # Should not raise any errors
        predicted_class, confidence = probe.score(logits)

        # Check output types and ranges
        assert isinstance(predicted_class, (int, float))
        assert predicted_class in [-1.0, 0.0, 1.0]
        assert isinstance(confidence, (int, float))
        assert 0.0 <= confidence <= 1.0

    def test_probabilities_normalization(self):
        """Test that probabilities are properly normalized.

        When answer tokens have significant probability mass, the returned
        probabilities should sum close to 1.0. When most probability is in
        other tokens (the "else" category), the sum can be less than 1.0.
        """
        cfg = {"name": "zeroshot"}
        model_cfg = {"model": {"name": "llama-base"}}
        io = Mock()

        probe = ZeroShotProbe(cfg=cfg, model_cfg=model_cfg, io=io)

        # Mock tokenizer with known token IDs
        def encode_fn(text, add_special_tokens=False):
            encoding_map = {
                "1": [16],
                " 1": [352],
                "2": [17],
                " 2": [353],
                "3": [18],
                " 3": [354],
                "4": [19],
                " 4": [355],
                "5": [20],
                " 5": [356],
                "6": [21],
                " 6": [357],
            }
            return encoding_map.get(text, [0])

        tokenizer = Mock()
        tokenizer.encode = Mock(side_effect=encode_fn)
        probe.set_tokenizer(tokenizer)

        # Test case 1: High probability on answer tokens
        vocab_size = 500
        logits = np.zeros((1, vocab_size), dtype=np.float32)
        logits[0, 16] = 10.0  # '1'
        logits[0, 352] = 9.0  # ' 1'
        logits[0, 17] = 5.0  # '2'
        logits[0, 353] = 4.0  # ' 2'

        probs = probe._collect_probabilities(logits)
        total_prob = sum(probs.values())

        # When answer tokens dominate, probabilities should sum close to 1.0
        assert np.isclose(
            total_prob, 1.0, atol=1e-5
        ), f"Expected probabilities to sum to ~1.0, got {total_prob}"

        # Test case 2: Low probability on answer tokens (most in "else")
        logits2 = np.ones((1, vocab_size), dtype=np.float32) * 5.0
        for tid in [16, 17, 18, 19, 20, 21, 352, 353, 354, 355, 356, 357]:
            logits2[0, tid] = 0.0  # Low logits for answer tokens

        probs2 = probe._collect_probabilities(logits2)
        total_prob2 = sum(probs2.values())

        # When "else" dominates, answer probabilities should be very low
        assert (
            total_prob2 < 0.1
        ), f"Expected very low total probability when else dominates, got {total_prob2}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
