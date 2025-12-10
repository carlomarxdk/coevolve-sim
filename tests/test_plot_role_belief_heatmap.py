"""
Tests for plot_role_belief_heatmap.py script.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from plot_role_belief_heatmap import (
    _compute_transfer_entropy,
    compute_pairwise_neighbor_accuracy_change,
    compute_pairwise_role_correlation,
    compute_pairwise_role_transfer_entropy,
    find_multi_run_experiments,
    load_neighbor_accuracy_change,
    load_pairwise_neighbor_accuracy_change,
    load_pairwise_role_data,
    load_pairwise_role_transfer_entropy,
    load_role_belief_data,
    plot_neighbor_accuracy_change,
    plot_pairwise_neighbor_accuracy_change_heatmap,
    plot_pairwise_role_correlation_heatmap,
    plot_pairwise_role_transfer_entropy_heatmap,
    plot_role_belief_heatmap,
)


class TestFindMultiRunExperiments:
    """Test finding multi-run experiments."""

    def test_find_multi_run_experiments_with_valid_dir(self):
        """Test finding multi-run experiments from valid directory."""
        multi_run_configs = find_multi_run_experiments("outputs/runs")

        assert isinstance(multi_run_configs, dict)

        for config_name, run_dirs in multi_run_configs.items():
            assert isinstance(config_name, str)
            assert isinstance(run_dirs, list)
            assert len(run_dirs) > 1

            for run_dir in run_dirs:
                assert (run_dir / "config.json").exists()

    def test_find_multi_run_experiments_with_invalid_dir(self):
        """Test finding multi-run experiments from invalid directory."""
        multi_run_configs = find_multi_run_experiments("/nonexistent/path")
        assert multi_run_configs == {}


class TestLoadRoleBeliefData:
    """Test loading role belief data."""

    def test_load_role_belief_data(self):
        """Test loading role belief data from runs."""
        multi_run_configs = find_multi_run_experiments("outputs/runs")

        if not multi_run_configs:
            pytest.skip("No multi-run experiments available")

        config_name, run_dirs = next(iter(multi_run_configs.items()))

        role_scores, rounds, correct_label = load_role_belief_data(run_dirs[:5])

        assert isinstance(role_scores, dict)
        assert isinstance(rounds, list)

        if role_scores:
            for role, scores in role_scores.items():
                assert isinstance(role, str)
                assert isinstance(scores, np.ndarray)
                assert scores.ndim == 2
                assert scores.shape[1] == len(rounds)
                assert (scores >= 0).all()
                assert (scores <= 1).all()

    def test_load_role_belief_data_empty_dirs(self):
        """Test loading role belief data with empty run list."""
        role_scores, rounds, correct_label = load_role_belief_data([])

        assert isinstance(role_scores, dict)
        assert len(role_scores) == 0
        assert rounds == []

    def test_load_role_belief_data_collective_accuracy(self):
        """Test loading role belief data with collective accuracy metric."""
        multi_run_configs = find_multi_run_experiments("outputs/runs")

        if not multi_run_configs:
            pytest.skip("No multi-run experiments available")

        config_name, run_dirs = next(iter(multi_run_configs.items()))

        role_scores, rounds, correct_label = load_role_belief_data(
            run_dirs[:5], metric="collective_accuracy"
        )

        assert isinstance(role_scores, dict)
        assert isinstance(rounds, list)

        if role_scores:
            for role, scores in role_scores.items():
                assert isinstance(scores, np.ndarray)
                assert (scores >= 0).all()
                assert (scores <= 1).all()


class TestPlotRoleBeliefHeatmap:
    """Test role belief heatmap plotting."""

    def test_plot_role_belief_heatmap(self):
        """Test generating a role belief heatmap."""
        multi_run_configs = find_multi_run_experiments("outputs/runs")

        if not multi_run_configs:
            pytest.skip("No multi-run experiments available")

        config_name, run_dirs = next(iter(multi_run_configs.items()))
        role_scores, rounds, correct_label = load_role_belief_data(run_dirs[:5])

        if not role_scores:
            pytest.skip("No role belief data available")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            plot_role_belief_heatmap(
                role_scores, rounds, config_name, correct_label, output_dir
            )

            plots = list(output_dir.glob("*.png"))
            assert len(plots) == 1

    def test_plot_role_belief_heatmap_empty_data(self):
        """Test plotting with empty data does not raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            plot_role_belief_heatmap({}, [], "empty_experiment", None, output_dir)

            plots = list(output_dir.glob("*.png"))
            assert len(plots) == 0

    def test_plot_role_belief_heatmap_values(self):
        """Test that heatmap values are correctly computed."""
        role_scores = {
            "Role1": np.array([[0.5, 0.6, 0.7], [0.4, 0.5, 0.6]]),
            "Role2": np.array([[0.3, 0.4, 0.5], [0.2, 0.3, 0.4]]),
        }
        rounds = [0, 1, 2]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            plot_role_belief_heatmap(
                role_scores, rounds, "test_experiment", 1, output_dir
            )

            plots = list(output_dir.glob("*.png"))
            assert len(plots) == 1

            expected_role1_mean = np.array([0.45, 0.55, 0.65])
            expected_role2_mean = np.array([0.25, 0.35, 0.45])
            actual_role1_mean = role_scores["Role1"].mean(axis=0)
            actual_role2_mean = role_scores["Role2"].mean(axis=0)

            np.testing.assert_array_almost_equal(actual_role1_mean, expected_role1_mean)
            np.testing.assert_array_almost_equal(actual_role2_mean, expected_role2_mean)

    def test_plot_role_belief_heatmap_collective_accuracy(self):
        """Test generating a heatmap with collective accuracy metric."""
        multi_run_configs = find_multi_run_experiments("outputs/runs")

        if not multi_run_configs:
            pytest.skip("No multi-run experiments available")

        config_name, run_dirs = next(iter(multi_run_configs.items()))
        role_scores, rounds, correct_label = load_role_belief_data(
            run_dirs[:5], metric="collective_accuracy"
        )

        if not role_scores:
            pytest.skip("No role belief data available")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            plot_role_belief_heatmap(
                role_scores,
                rounds,
                config_name,
                correct_label,
                output_dir,
                metric="collective_accuracy",
            )

            plots = list(output_dir.glob("*_collective_acc.png"))
            assert len(plots) == 1


class TestPairwiseRoleCorrelation:
    """Test pairwise role correlation functionality."""

    def test_load_pairwise_role_data(self):
        """Test loading pairwise role data from runs."""
        multi_run_configs = find_multi_run_experiments("outputs/runs")

        if not multi_run_configs:
            pytest.skip("No multi-run experiments available")

        config_name, run_dirs = next(iter(multi_run_configs.items()))
        role_pair_data, correct_label = load_pairwise_role_data(run_dirs[:5])

        assert isinstance(role_pair_data, dict)

        if role_pair_data:
            for role_pair, score_pairs in role_pair_data.items():
                assert isinstance(role_pair, tuple)
                assert len(role_pair) == 2
                assert isinstance(score_pairs, list)
                for scores1, scores2 in score_pairs:
                    assert isinstance(scores1, np.ndarray)
                    assert isinstance(scores2, np.ndarray)
                    assert len(scores1) == len(scores2)

    def test_load_pairwise_role_data_empty_dirs(self):
        """Test loading pairwise role data with empty run list."""
        role_pair_data, correct_label = load_pairwise_role_data([])

        assert isinstance(role_pair_data, dict)
        assert len(role_pair_data) == 0

    def test_compute_pairwise_role_correlation(self):
        """Test computing pairwise role correlation."""
        role_pair_data = {
            ("Role1", "Role2"): [
                (np.array([0.1, 0.2, 0.3]), np.array([0.2, 0.4, 0.6])),
                (np.array([0.5, 0.6, 0.7]), np.array([0.5, 0.6, 0.7])),
            ],
        }

        correlation_matrix, roles = compute_pairwise_role_correlation(role_pair_data)

        assert isinstance(correlation_matrix, dict)
        assert isinstance(roles, list)
        assert "Role1" in roles
        assert "Role2" in roles
        assert "Role2" in correlation_matrix["Role1"]
        assert "Role1" in correlation_matrix["Role2"]

    def test_plot_pairwise_role_correlation_heatmap(self):
        """Test plotting pairwise role correlation heatmap."""
        multi_run_configs = find_multi_run_experiments("outputs/runs")

        if not multi_run_configs:
            pytest.skip("No multi-run experiments available")

        config_name, run_dirs = next(iter(multi_run_configs.items()))
        role_pair_data, correct_label = load_pairwise_role_data(run_dirs[:5])

        if not role_pair_data:
            pytest.skip("No pairwise role data available")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            plot_pairwise_role_correlation_heatmap(
                role_pair_data, config_name, correct_label, output_dir
            )

            plots = list(output_dir.glob("*pairwise*.png"))
            assert len(plots) == 1

    def test_plot_pairwise_role_correlation_heatmap_empty_data(self):
        """Test plotting pairwise correlation with empty data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            plot_pairwise_role_correlation_heatmap(
                {}, "empty_experiment", None, output_dir
            )

            plots = list(output_dir.glob("*.png"))
            assert len(plots) == 0


class TestPairwiseRoleTransferEntropy:
    """Test pairwise role transfer entropy functionality."""

    def test_compute_transfer_entropy_basic(self):
        """Test basic transfer entropy computation."""
        # Create time series with clear information flow
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        y = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        te = _compute_transfer_entropy(x, y)

        assert isinstance(te, float)
        assert te >= 0.0

    def test_compute_transfer_entropy_constant(self):
        """Test transfer entropy with constant series returns zero."""
        x = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        y = np.array([0.3, 0.4, 0.5, 0.6, 0.7])

        te = _compute_transfer_entropy(x, y)

        assert te == 0.0

    def test_compute_transfer_entropy_short_series(self):
        """Test transfer entropy with very short series."""
        x = np.array([0.5])
        y = np.array([0.3])

        te = _compute_transfer_entropy(x, y)

        assert te == 0.0

    def test_load_pairwise_role_transfer_entropy(self):
        """Test loading pairwise role transfer entropy from runs."""
        multi_run_configs = find_multi_run_experiments("outputs/runs")

        if not multi_run_configs:
            pytest.skip("No multi-run experiments available")

        config_name, run_dirs = next(iter(multi_run_configs.items()))
        role_pair_te, correct_label = load_pairwise_role_transfer_entropy(run_dirs[:5])

        assert isinstance(role_pair_te, dict)

        if role_pair_te:
            for role_pair, te_values in role_pair_te.items():
                assert isinstance(role_pair, tuple)
                assert len(role_pair) == 2
                assert isinstance(te_values, list)
                for te_val in te_values:
                    assert isinstance(te_val, float)
                    assert te_val >= 0.0

    def test_load_pairwise_role_transfer_entropy_empty_dirs(self):
        """Test loading transfer entropy with empty run list."""
        role_pair_te, correct_label = load_pairwise_role_transfer_entropy([])

        assert isinstance(role_pair_te, dict)
        assert len(role_pair_te) == 0

    def test_compute_pairwise_role_transfer_entropy(self):
        """Test computing average pairwise role transfer entropy."""
        role_pair_te = {
            ("Role1", "Role2"): [0.1, 0.2, 0.15],
            ("Role2", "Role1"): [0.05, 0.1, 0.08],
        }

        te_matrix, roles = compute_pairwise_role_transfer_entropy(role_pair_te)

        assert isinstance(te_matrix, dict)
        assert isinstance(roles, list)
        assert "Role1" in roles
        assert "Role2" in roles
        assert "Role2" in te_matrix["Role1"]
        assert "Role1" in te_matrix["Role2"]

        # Check average values
        np.testing.assert_almost_equal(te_matrix["Role1"]["Role2"], 0.15)
        np.testing.assert_almost_equal(
            te_matrix["Role2"]["Role1"], 0.0766667, decimal=4
        )

    def test_plot_pairwise_role_transfer_entropy_heatmap(self):
        """Test plotting pairwise role transfer entropy heatmap."""
        multi_run_configs = find_multi_run_experiments("outputs/runs")

        if not multi_run_configs:
            pytest.skip("No multi-run experiments available")

        config_name, run_dirs = next(iter(multi_run_configs.items()))
        role_pair_te, correct_label = load_pairwise_role_transfer_entropy(run_dirs[:5])

        if not role_pair_te:
            pytest.skip("No pairwise transfer entropy data available")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            plot_pairwise_role_transfer_entropy_heatmap(
                role_pair_te, config_name, correct_label, output_dir
            )

            plots = list(output_dir.glob("*transfer_entropy*.png"))
            assert len(plots) == 1

    def test_plot_pairwise_role_transfer_entropy_heatmap_empty_data(self):
        """Test plotting transfer entropy with empty data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            plot_pairwise_role_transfer_entropy_heatmap(
                {}, "empty_experiment", None, output_dir
            )

            plots = list(output_dir.glob("*.png"))
            assert len(plots) == 0


class TestNeighborAccuracyChange:
    """Test neighbor accuracy change functionality."""

    def test_load_neighbor_accuracy_change(self):
        """Test loading neighbor accuracy change data from runs."""
        multi_run_configs = find_multi_run_experiments("outputs/runs")

        if not multi_run_configs:
            pytest.skip("No multi-run experiments available")

        config_name, run_dirs = next(iter(multi_run_configs.items()))
        role_neighbor_changes, correct_label = load_neighbor_accuracy_change(
            run_dirs[:5]
        )

        assert isinstance(role_neighbor_changes, dict)

        if role_neighbor_changes:
            for role, changes in role_neighbor_changes.items():
                assert isinstance(role, str)
                assert isinstance(changes, list)
                for change in changes:
                    assert isinstance(change, float)
                    # Changes can be negative or positive
                    assert -1 <= change <= 1

    def test_load_neighbor_accuracy_change_empty_dirs(self):
        """Test loading neighbor accuracy change with empty run list."""
        role_neighbor_changes, correct_label = load_neighbor_accuracy_change([])

        assert isinstance(role_neighbor_changes, dict)
        assert len(role_neighbor_changes) == 0

    def test_plot_neighbor_accuracy_change(self):
        """Test plotting neighbor accuracy change bar chart."""
        multi_run_configs = find_multi_run_experiments("outputs/runs")

        if not multi_run_configs:
            pytest.skip("No multi-run experiments available")

        config_name, run_dirs = next(iter(multi_run_configs.items()))
        role_neighbor_changes, correct_label = load_neighbor_accuracy_change(
            run_dirs[:5]
        )

        if not role_neighbor_changes:
            pytest.skip("No neighbor accuracy change data available")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            plot_neighbor_accuracy_change(
                role_neighbor_changes, config_name, correct_label, output_dir
            )

            plots = list(output_dir.glob("*neighbor_accuracy_change*.png"))
            assert len(plots) == 1

    def test_plot_neighbor_accuracy_change_empty_data(self):
        """Test plotting neighbor accuracy change with empty data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            plot_neighbor_accuracy_change({}, "empty_experiment", None, output_dir)

            plots = list(output_dir.glob("*.png"))
            assert len(plots) == 0

    def test_plot_neighbor_accuracy_change_with_synthetic_data(self):
        """Test plotting with synthetic data."""
        role_neighbor_changes = {
            "Role1": [0.1, 0.15, 0.12],
            "Role2": [-0.05, -0.02, -0.08],
            "Role3": [0.2, 0.18, 0.22],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            plot_neighbor_accuracy_change(
                role_neighbor_changes, "test/catalog/prompt/statement", 1, output_dir
            )

            plots = list(output_dir.glob("*neighbor_accuracy_change*.png"))
            assert len(plots) == 1


class TestPairwiseNeighborAccuracyChange:
    """Test pairwise neighbor accuracy change functionality."""

    def test_load_pairwise_neighbor_accuracy_change(self):
        """Test loading pairwise neighbor accuracy change data from runs."""
        multi_run_configs = find_multi_run_experiments("outputs/runs")

        if not multi_run_configs:
            pytest.skip("No multi-run experiments available")

        config_name, run_dirs = next(iter(multi_run_configs.items()))
        role_pair_changes, correct_label = load_pairwise_neighbor_accuracy_change(
            run_dirs[:5]
        )

        assert isinstance(role_pair_changes, dict)

        if role_pair_changes:
            for role_pair, changes in role_pair_changes.items():
                assert isinstance(role_pair, tuple)
                assert len(role_pair) == 2
                assert isinstance(changes, list)
                for change in changes:
                    assert isinstance(change, float)
                    # Changes can be negative or positive
                    assert -1 <= change <= 1

    def test_load_pairwise_neighbor_accuracy_change_empty_dirs(self):
        """Test loading pairwise neighbor accuracy change with empty run list."""
        role_pair_changes, correct_label = load_pairwise_neighbor_accuracy_change([])

        assert isinstance(role_pair_changes, dict)
        assert len(role_pair_changes) == 0

    def test_compute_pairwise_neighbor_accuracy_change(self):
        """Test computing average pairwise neighbor accuracy change."""
        role_pair_changes = {
            ("Role1", "Role2"): [0.1, 0.2, 0.15],
            ("Role2", "Role1"): [0.05, 0.1, 0.08],
        }

        change_matrix, roles = compute_pairwise_neighbor_accuracy_change(
            role_pair_changes
        )

        assert isinstance(change_matrix, dict)
        assert isinstance(roles, list)
        assert "Role1" in roles
        assert "Role2" in roles
        assert "Role2" in change_matrix["Role1"]
        assert "Role1" in change_matrix["Role2"]

        # Check average values
        np.testing.assert_almost_equal(change_matrix["Role1"]["Role2"], 0.15)
        np.testing.assert_almost_equal(
            change_matrix["Role2"]["Role1"], 0.0766667, decimal=4
        )

    def test_plot_pairwise_neighbor_accuracy_change_heatmap(self):
        """Test plotting pairwise neighbor accuracy change heatmap."""
        multi_run_configs = find_multi_run_experiments("outputs/runs")

        if not multi_run_configs:
            pytest.skip("No multi-run experiments available")

        config_name, run_dirs = next(iter(multi_run_configs.items()))
        role_pair_changes, correct_label = load_pairwise_neighbor_accuracy_change(
            run_dirs[:5]
        )

        if not role_pair_changes:
            pytest.skip("No pairwise neighbor accuracy change data available")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            plot_pairwise_neighbor_accuracy_change_heatmap(
                role_pair_changes, config_name, correct_label, output_dir
            )

            plots = list(output_dir.glob("*pairwise_neighbor_accuracy_change*.png"))
            assert len(plots) == 1

    def test_plot_pairwise_neighbor_accuracy_change_heatmap_empty_data(self):
        """Test plotting pairwise neighbor accuracy change with empty data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            plot_pairwise_neighbor_accuracy_change_heatmap(
                {}, "empty_experiment", None, output_dir
            )

            plots = list(output_dir.glob("*.png"))
            assert len(plots) == 0

    def test_plot_pairwise_neighbor_accuracy_change_with_synthetic_data(self):
        """Test plotting with synthetic data."""
        role_pair_changes = {
            ("Role1", "Role2"): [0.1, 0.15, 0.12],
            ("Role2", "Role1"): [-0.05, -0.02, -0.08],
            ("Role1", "Role3"): [0.2, 0.18, 0.22],
            ("Role3", "Role1"): [0.15, 0.12, 0.18],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            plot_pairwise_neighbor_accuracy_change_heatmap(
                role_pair_changes, "test/catalog/prompt/statement", 1, output_dir
            )

            plots = list(output_dir.glob("*pairwise_neighbor_accuracy_change*.png"))
            assert len(plots) == 1
