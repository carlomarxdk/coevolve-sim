"""
Tests for ZeroShotProbe class
"""

from unittest.mock import Mock

import numpy as np
import pytest

from src.core.probe import ZeroShotProbe


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
        assert len(probe.token_ids) == 3  # Should have token IDs for 1-3

        # Check that each option has token IDs
        for option in ["1", "2", "3"]:
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

        predicted_class, p_true, final_scores = probe.score(logits)

        assert predicted_class == 1.0  # Should predict True
        assert p_true > 0.5  # P(true) should be high
        assert final_scores is not None  # Should have complete scores
        assert len(final_scores) == 3  # [P(true), P(false), P(uncertain)]
        assert final_scores[0] > 0.5  # P(true) should be high

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

        predicted_class, p_true, scores = probe.score(logits)

        assert predicted_class == 0.0  # Should predict False
        assert p_true < 0.5  # P(true) should be low when predicting False
        assert scores[1] > 0.5  # P(false) should be high

    def test_score_with_uncertain_prediction(
        self, basic_config, model_config, mock_io, mock_tokenizer
    ):
        """Test scoring when model predicts Uncertain (option 3)"""
        probe = ZeroShotProbe(cfg=basic_config, model_cfg=model_config, io=mock_io)
        probe.set_tokenizer(mock_tokenizer)

        # Create logits favoring token 18 (option '3')
        vocab_size = 500
        logits = np.zeros((1, vocab_size), dtype=np.float32)
        logits[0, 18] = 10.0  # High logit for token '3'
        logits[0, 354] = 9.0  # High logit for ' 3'

        predicted_class, p_true, final_scores = probe.score(logits)

        assert predicted_class == -1.0  # Should predict Uncertain
        assert p_true < 0.5  # P(true) should be low when predicting Uncertain
        assert final_scores is not None  # Should have complete scores
        assert len(final_scores) == 3  # [P(true), P(false), P(uncertain)]
        assert final_scores[2] > 0.5  # P(uncertain) should be high

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

        predicted_class, confidence, scores = probe.score(logits)

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
        predicted_class, confidence, scores = probe.score(logits)

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
        for tid in [16, 17, 18, 352, 353, 354]:
            logits2[0, tid] = 0.0  # Low logits for answer tokens

        probs2 = probe._collect_probabilities(logits2)
        # Exclude "else" from the sum to check only answer token probabilities
        answer_tokens_prob_sum = sum(v for k, v in probs2.items() if k != "else")

        # When "else" dominates, answer probabilities should be very low
        assert (
            answer_tokens_prob_sum < 0.1
        ), f"Expected very low answer probability when else dominates, got {answer_tokens_prob_sum}"

    def test_complete_scores_structure(self):
        """Test that complete_scores returns the correct structure."""
        cfg = {"name": "zeroshot"}
        model_cfg = {"model": {"name": "llama-base"}}
        io = Mock()

        probe = ZeroShotProbe(cfg=cfg, model_cfg=model_cfg, io=io)

        # Mock tokenizer
        def encode_fn(text, add_special_tokens=False):
            encoding_map = {
                "1": [16],
                " 1": [352],
                "2": [17],
                " 2": [353],
                "3": [18],
                " 3": [354],
            }
            return encoding_map.get(text, [0])

        tokenizer = Mock()
        tokenizer.encode = Mock(side_effect=encode_fn)
        probe.set_tokenizer(tokenizer)

        # Test with various predictions
        vocab_size = 500

        # Case 1: Favor True
        logits1 = np.zeros((1, vocab_size), dtype=np.float32)
        logits1[0, 16] = 10.0
        _, _, scores1 = probe.score(logits1)

        assert isinstance(scores1, list)
        assert len(scores1) == 3  # [P(true), P(false), P(uncertain)]
        assert scores1[0] > scores1[1]  # P(true) > P(false)
        assert scores1[0] > scores1[2]  # P(true) > P(uncertain)
        assert np.isclose(sum(scores1), 1.0, atol=1e-5)  # Should sum to ~1.0

        # Case 2: Favor False
        logits2 = np.zeros((1, vocab_size), dtype=np.float32)
        logits2[0, 17] = 10.0
        _, _, scores2 = probe.score(logits2)

        assert scores2[1] > scores2[0]  # P(false) > P(true)
        assert scores2[1] > scores2[2]  # P(false) > P(uncertain)

        # Case 3: Favor Uncertain
        logits3 = np.zeros((1, vocab_size), dtype=np.float32)
        logits3[0, 18] = 10.0
        _, _, scores3 = probe.score(logits3)

        assert scores3[2] > scores3[0]  # P(uncertain) > P(true)
        assert scores3[2] > scores3[1]  # P(uncertain) > P(false)

    def test_complete_scores_consistency_with_prediction(self):
        """Test that complete_scores is consistent with predicted class."""
        cfg = {"name": "zeroshot"}
        model_cfg = {"model": {"name": "llama-base"}}
        io = Mock()

        probe = ZeroShotProbe(cfg=cfg, model_cfg=model_cfg, io=io)

        # Mock tokenizer
        def encode_fn(text, add_special_tokens=False):
            encoding_map = {
                "1": [16],
                " 1": [352],
                "2": [17],
                " 2": [353],
                "3": [18],
                " 3": [354],
            }
            return encoding_map.get(text, [0])

        tokenizer = Mock()
        tokenizer.encode = Mock(side_effect=encode_fn)
        probe.set_tokenizer(tokenizer)

        vocab_size = 500

        # Test that predicted class matches the highest probability in complete_scores
        for token_id, expected_class, expected_max_idx in [
            (16, 1.0, 0),   # Token '1' -> True
            (17, 0.0, 1),   # Token '2' -> False
            (18, -1.0, 2),  # Token '3' -> Uncertain
        ]:
            logits = np.zeros((1, vocab_size), dtype=np.float32)
            logits[0, token_id] = 10.0

            predicted_class, p_true, complete_scores = probe.score(logits)

            assert predicted_class == expected_class
            assert np.argmax(complete_scores) == expected_max_idx
            assert complete_scores[0] == p_true  # First element should match p_true


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
