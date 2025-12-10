"""
Comprehensive tests for general metrics functions
"""

import numpy as np
import pytest

from src.metrics.general import (
    belief_change_metrics,
    binary_entropy,
    binary_entropy_normalized,
    binary_magnetization,
    binary_polarization,
    compute_diffusion_speed,
    consensus_fraction,
    diffusion_speed,
    entropy_multiclass_normalized,
    granger_influence_score,
    influence_matrix,
    information_leadership,
    label_distribution,
    mutual_information_matrix,
    simple_influence_scores,
    transfer_entropy_matrix,
)


class TestLabelDistribution:
    """Test label distribution function"""

    def test_label_distribution_basic(self):
        """Test basic label distribution computation"""
        # 2 runs, 4 agents
        labels = np.array([[1, 0, 1, -1], [1, 1, 0, -1]])
        avg_fractions, std_fractions = label_distribution(labels)

        # Check shape
        assert avg_fractions.shape == (3,)
        assert std_fractions.shape == (3,)

        # Check values for label counts: [0, 1, -1]
        # Run 1: [1, 2, 1], Run 2: [1, 2, 1] => avg = [0.25, 0.5, 0.25]
        assert np.allclose(avg_fractions, [0.25, 0.5, 0.25])
        assert np.allclose(std_fractions, [0, 0, 0])


class TestDiffusionSpeed:
    """Test diffusion speed functions"""

    def test_diffusion_speed_categorical(self):
        """Test diffusion speed for categorical labels"""
        # 3 rounds, 3 agents
        labels = np.array([[1, 0, 1], [1, 1, 1], [0, 1, 1]])

        speed = diffusion_speed(labels, categories=True)

        # Round 0->1: agent 1 changes (1/3)
        # Round 1->2: agent 0 changes (1/3)
        assert len(speed) == 2
        assert np.allclose(speed, [1 / 3, 1 / 3])

    def test_diffusion_speed_continuous(self):
        """Test diffusion speed for continuous values"""
        # 3 rounds, 2 agents
        labels = np.array([[0.0, 0.5], [0.3, 0.5], [0.3, 0.8]])

        speed = diffusion_speed(labels, categories=False)

        # Round 0->1: changes [0.3, 0] => avg = 0.15
        # Round 1->2: changes [0, 0.3] => avg = 0.15
        assert len(speed) == 2
        assert np.allclose(speed, [0.15, 0.15])

    def test_compute_diffusion_speed_multiple_runs(self):
        """Test diffusion speed across multiple runs"""
        # 2 runs, 3 rounds, 2 agents
        labels = np.array([[[1, 0], [1, 1], [0, 1]], [[0, 1], [1, 1], [1, 0]]])

        speed = compute_diffusion_speed(labels, categories=True)

        # Shape should be (2, 2) - 2 runs, 2 transitions
        assert speed.shape == (2, 2)


class TestConsensusFraction:
    """Test consensus fraction function"""

    def test_consensus_fraction_majority(self):
        """Test with clear majority"""
        labels = np.array([1, 1, 1, 0])
        frac = consensus_fraction(labels)
        assert frac == 0.75  # 3/4

    def test_consensus_fraction_tied(self):
        """Test with tied beliefs"""
        labels = np.array([1, 1, 0, 0])
        frac = consensus_fraction(labels)
        assert frac == 0.5  # 2/4

    def test_consensus_fraction_unanimous(self):
        """Test with unanimous beliefs"""
        labels = np.array([1, 1, 1, 1])
        frac = consensus_fraction(labels)
        assert frac == 1.0

    def test_consensus_fraction_empty(self):
        """Test with empty array"""
        labels = np.array([])
        frac = consensus_fraction(labels)
        assert frac == 0.0


class TestBeliefChangeMetrics:
    """Test belief change metrics"""

    def test_belief_change_no_previous(self):
        """Test when there's no previous round"""
        curr = np.array([1, 0, 1])
        flip_rate, l1, l2 = belief_change_metrics(None, curr)

        assert flip_rate == 0.0
        assert l1 == 0.0
        assert l2 == 0.0

    def test_belief_change_with_flips(self):
        """Test with some belief changes"""
        prev = np.array([1, 0, 1, 0])
        curr = np.array([1, 1, 0, 0])

        flip_rate, l1, l2 = belief_change_metrics(prev, curr)

        # 2 out of 4 agents changed
        assert flip_rate == 0.5
        # L1: |0| + |1| + |-1| + |0| = 2
        assert l1 == 2.0
        # L2: sqrt(0^2 + 1^2 + 1^2 + 0^2) = sqrt(2)
        assert np.isclose(l2, np.sqrt(2))


class TestBinaryMetrics:
    """Test binary belief metrics"""

    def test_binary_magnetization(self):
        """Test magnetization computation"""
        # All agree (1)
        beliefs = np.array([1, 1, 1])
        mag = binary_magnetization(beliefs)
        assert mag == 1.0

        # All disagree (0)
        beliefs = np.array([0, 0, 0])
        mag = binary_magnetization(beliefs)
        assert mag == 1.0

        # 50-50 split
        beliefs = np.array([1, 1, 0, 0])
        mag = binary_magnetization(beliefs)
        assert mag == 0.0

    def test_binary_polarization(self):
        """Test polarization computation"""
        # All agree
        beliefs = np.array([1, 1, 1])
        pol = binary_polarization(beliefs)
        assert pol == 0.0

        # 50-50 split (maximum polarization)
        beliefs = np.array([1, 1, 0, 0])
        pol = binary_polarization(beliefs)
        assert pol == 1.0

        # 75-25 split
        beliefs = np.array([1, 1, 1, 0])
        pol = binary_polarization(beliefs)
        # P = 4 * 0.75 * 0.25 = 0.75
        assert np.isclose(pol, 0.75)

    def test_binary_entropy(self):
        """Test binary entropy"""
        # All agree (minimum entropy)
        beliefs = np.array([1, 1, 1])
        ent = binary_entropy(beliefs)
        assert np.isclose(ent, 0.0)

        # 50-50 split (maximum entropy)
        beliefs = np.array([1, 1, 0, 0])
        ent = binary_entropy(beliefs)
        # H = -0.5*ln(0.5) - 0.5*ln(0.5) = ln(2)
        assert np.isclose(ent, np.log(2))

    def test_binary_entropy_normalized(self):
        """Test normalized binary entropy"""
        # Maximum entropy (50-50 split)
        beliefs = np.array([1, 1, 0, 0])
        ent = binary_entropy_normalized(beliefs)
        assert np.isclose(ent, 1.0)

        # Minimum entropy (all agree)
        beliefs = np.array([1, 1, 1])
        ent = binary_entropy_normalized(beliefs)
        assert np.isclose(ent, 0.0)


class TestMulticlassEntropy:
    """Test multiclass entropy"""

    def test_multiclass_entropy_uniform(self):
        """Test with uniform distribution"""
        beliefs = np.array([0, 1, 2])
        ent = entropy_multiclass_normalized(beliefs)
        # Maximum entropy for 3 classes
        assert np.isclose(ent, 1.0)

    def test_multiclass_entropy_concentrated(self):
        """Test with concentrated distribution"""
        beliefs = np.array([0, 0, 0])
        ent = entropy_multiclass_normalized(beliefs)
        # Minimum entropy
        assert np.isclose(ent, 0.0)


class TestInfluenceMetrics:
    """Test influence and information metrics"""

    def test_granger_influence_score_simple(self):
        """Test Granger influence with simple history"""
        # 3 time steps, 2 agents
        history = [np.array([0, 0]), np.array([1, 0]), np.array([1, 1])]

        scores = granger_influence_score(history)

        # Should return array of length 2
        assert len(scores) == 2
        assert isinstance(scores, np.ndarray)

    def test_granger_influence_score_insufficient_data(self):
        """Test with insufficient time steps"""
        # Only 1 time step
        history = [np.array([0, 0])]

        scores = granger_influence_score(history)

        # Should return zero array
        assert len(scores) == 2
        assert np.all(scores == 0)

    def test_simple_influence_scores(self):
        """Test simple correlation-based influence"""
        # 3 time steps, 2 agents
        history = [np.array([0, 0]), np.array([1, 0]), np.array([1, 1])]

        scores = simple_influence_scores(history)

        # Should return array of length 2
        assert len(scores) == 2
        assert isinstance(scores, np.ndarray)

    def test_simple_influence_scores_insufficient_data(self):
        """Test with insufficient data"""
        history = [np.array([0, 0])]

        scores = simple_influence_scores(history)

        # Should return empty array
        assert len(scores) == 0


class TestMutualInformation:
    """Test mutual information and transfer entropy"""

    def test_mutual_information_matrix(self):
        """Test MI matrix computation"""
        # 3 time steps, 3 agents
        history = [np.array([0, 0, 1]), np.array([1, 0, 1]), np.array([1, 1, 0])]

        mi = mutual_information_matrix(history)

        # Should be 3x3 matrix
        assert mi.shape == (3, 3)
        # Diagonal should be NaN
        assert np.all(np.isnan(np.diag(mi)))

    def test_mutual_information_all_same_beliefs(self):
        """Test MI matrix when all agents have the same belief"""
        # All agents agree on belief=1 across all time steps
        history = [np.ones(5) for _ in range(10)]

        mi = mutual_information_matrix(history)

        # Should be 5x5 matrix
        assert mi.shape == (5, 5)
        # Diagonal should be NaN
        assert np.all(np.isnan(np.diag(mi)))
        # Off-diagonal should be 0 (no mutual information when all are the same)
        assert np.all(mi[~np.isnan(mi)] == 0.0)
        # Should not have any negative values
        assert np.all(mi[~np.isnan(mi)] >= 0.0)

    def test_mutual_information_empty(self):
        """Test with empty history"""
        history = []

        mi = mutual_information_matrix(history)

        # Should return empty array
        assert mi.size == 0

    def test_transfer_entropy_matrix(self):
        """Test transfer entropy computation"""
        # 3 time steps, 2 agents
        history = [np.array([0, 0]), np.array([1, 0]), np.array([1, 1])]

        te = transfer_entropy_matrix(history)

        # Should be 2x2 matrix
        assert te.shape == (2, 2)
        # Diagonal should be NaN
        assert np.all(np.isnan(np.diag(te)))
        # Values should be non-negative
        te_no_nan = te[~np.isnan(te)]
        assert np.all(te_no_nan >= 0)

    def test_influence_matrix(self):
        """Test influence matrix from transfer entropy"""
        te = np.array([[np.nan, 0.5], [0.3, np.nan]])

        inf = influence_matrix(te)

        # Should have same shape
        assert inf.shape == te.shape
        # Diagonal should be 0
        assert np.all(np.diag(inf) == 0)
        # Off-diagonal should match (no NaN)
        assert inf[0, 1] == 0.5
        assert inf[1, 0] == 0.3

    def test_information_leadership(self):
        """Test leadership score computation"""
        te = np.array([[np.nan, 0.5, 0.3], [0.2, np.nan, 0.1], [0.0, 0.0, np.nan]])

        leadership = information_leadership(te)

        # Should be array of length 3
        assert len(leadership) == 3
        # Agent 0 influences most: 0.5 + 0.3 = 0.8
        assert np.isclose(leadership[0], 0.8)
        # Agent 1: 0.2 + 0.1 = 0.3
        assert np.isclose(leadership[1], 0.3)
        # Agent 2: 0.0 + 0.0 = 0.0
        assert np.isclose(leadership[2], 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
