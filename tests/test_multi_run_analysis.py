"""
Tests for multi-run experiment analysis tools.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from multi_run_analysis import (
    analyze_network_structure_impact,
    compute_belief_evolution_statistics,
    compute_diffusion_speed,
    find_multi_run_experiments,
    generate_comparison_report,
    load_multi_run_belief_data,
    plot_multi_run_belief_evolution,
    plot_network_impact_analysis,
)


class TestFindMultiRunExperiments:
    """Test finding multi-run experiments."""

    def test_find_multi_run_experiments_with_valid_dir(self):
        """Test finding multi-run experiments from valid directory."""
        multi_run_configs = find_multi_run_experiments("outputs/runs")

        # Should return a dictionary
        assert isinstance(multi_run_configs, dict)

        # Each value should be a list of paths
        for config_name, run_dirs in multi_run_configs.items():
            assert isinstance(config_name, str)
            assert isinstance(run_dirs, list)
            assert len(run_dirs) > 1  # Multi-run means more than 1

            # Each run dir should have a config.json
            for run_dir in run_dirs:
                assert (run_dir / "config.json").exists()

    def test_find_multi_run_experiments_with_invalid_dir(self):
        """Test finding multi-run experiments from invalid directory."""
        multi_run_configs = find_multi_run_experiments("/nonexistent/path")
        assert multi_run_configs == {}


class TestLoadMultiRunBeliefData:
    """Test loading belief data across multiple runs."""

    def test_load_multi_run_belief_data(self):
        """Test loading belief data from multiple runs."""
        # Find a multi-run configuration
        multi_run_configs = find_multi_run_experiments("outputs/runs")

        if not multi_run_configs:
            pytest.skip("No multi-run experiments available")

        # Get first configuration
        config_name, run_dirs = next(iter(multi_run_configs.items()))

        labels, scores, rounds = load_multi_run_belief_data(
            run_dirs[:5]
        )  # Use first 5 runs

        # Check return types
        assert isinstance(labels, np.ndarray)
        assert isinstance(scores, np.ndarray)
        assert isinstance(rounds, list)

        if labels.size > 0:
            # Check shapes
            assert labels.ndim == 3  # (runs, rounds, agents)
            assert scores.ndim == 3
            assert labels.shape == scores.shape
            assert labels.shape[1] == len(rounds)

    def test_load_multi_run_belief_data_empty_dirs(self):
        """Test loading belief data with no valid directories."""
        labels, scores, rounds = load_multi_run_belief_data([])

        assert labels.size == 0
        assert scores.size == 0
        assert rounds == []


class TestComputeBeliefEvolutionStatistics:
    """Test computing belief evolution statistics."""

    def test_compute_statistics_with_data(self):
        """Test computing statistics with valid data."""
        # Create dummy data
        num_runs, num_rounds, num_agents = 5, 10, 16
        labels = np.random.randn(num_runs, num_rounds, num_agents)
        scores = np.random.randn(num_runs, num_rounds, num_agents)

        stats = compute_belief_evolution_statistics(labels, scores)

        # Check that required keys are present
        assert "num_runs" in stats
        assert "num_rounds" in stats
        assert "num_agents" in stats
        assert "label_mean_per_round" in stats
        assert "label_std_per_round" in stats
        assert "score_mean_per_round" in stats
        assert "consensus_per_run" in stats

        # Check values
        assert stats["num_runs"] == num_runs
        assert stats["num_rounds"] == num_rounds
        assert stats["num_agents"] == num_agents
        assert len(stats["label_mean_per_round"]) == num_rounds
        assert len(stats["consensus_per_run"]) == num_runs

    def test_compute_statistics_empty_data(self):
        """Test computing statistics with empty data."""
        labels = np.array([])
        scores = np.array([])

        stats = compute_belief_evolution_statistics(labels, scores)

        assert stats == {}


class TestAnalyzeNetworkStructureImpact:
    """Test network structure impact analysis."""

    def test_analyze_network_structure_impact(self):
        """Test analyzing network structure impact."""
        # Find a multi-run configuration
        multi_run_configs = find_multi_run_experiments("outputs/runs")

        if not multi_run_configs:
            pytest.skip("No multi-run experiments available")

        # Get first configuration
        config_name, run_dirs = next(iter(multi_run_configs.items()))

        df = analyze_network_structure_impact(run_dirs[:5])

        # Should return a DataFrame
        import pandas as pd

        assert isinstance(df, pd.DataFrame)

        if not df.empty:
            # Check that expected columns are present
            expected_cols = [
                "run_id",
                "seed",
                "num_nodes",
                "num_edges",
                "density",
                "final_consensus",
            ]
            for col in expected_cols:
                assert col in df.columns

    def test_analyze_network_structure_impact_empty(self):
        """Test analyzing network structure with empty run list."""
        import pandas as pd

        df = analyze_network_structure_impact([])

        assert isinstance(df, pd.DataFrame)
        assert df.empty


class TestComputeDiffusionSpeed:
    """Test diffusion speed computation."""

    def test_compute_diffusion_speed(self):
        """Test computing diffusion speed."""
        # Create dummy data with known changes
        num_runs, num_rounds, num_agents = 3, 5, 10
        labels = np.zeros((num_runs, num_rounds, num_agents))

        # Add some changes
        labels[:, 1:, :] = 1.0  # All agents change to 1 after first round

        speed = compute_diffusion_speed(labels)

        assert speed.shape == (num_runs, num_rounds - 1)
        # First transition should show maximum change
        assert speed[:, 0].mean() > 0

    def test_compute_diffusion_speed_insufficient_rounds(self):
        """Test computing diffusion speed with insufficient rounds."""
        labels = np.zeros((3, 1, 10))  # Only 1 round

        speed = compute_diffusion_speed(labels)

        assert speed.size == 0


class TestPlottingFunctions:
    """Test plotting functions."""

    def test_plot_multi_run_belief_evolution(self):
        """Test plotting belief evolution."""
        # Create dummy data
        num_runs, num_rounds, num_agents = 3, 10, 16
        labels = np.random.randn(num_runs, num_rounds, num_agents)
        scores = np.random.randn(num_runs, num_rounds, num_agents)
        rounds = list(range(num_rounds))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Should not raise exception
            plot_multi_run_belief_evolution(
                labels, scores, rounds, "test_config", output_dir
            )

            # Check that plot was created
            plots = list(output_dir.glob("*.png"))
            assert len(plots) == 1

    def test_plot_network_impact_analysis(self):
        """Test plotting network impact analysis."""
        # Find a multi-run configuration
        multi_run_configs = find_multi_run_experiments("outputs/runs")

        if not multi_run_configs:
            pytest.skip("No multi-run experiments available")

        # Get first configuration
        config_name, run_dirs = next(iter(multi_run_configs.items()))

        df = analyze_network_structure_impact(run_dirs[:5])

        if df.empty:
            pytest.skip("No network data available")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Should not raise exception
            plot_network_impact_analysis(df, "test_config", output_dir)

            # Check that plot was created
            plots = list(output_dir.glob("*.png"))
            assert len(plots) == 1

    def test_plot_multi_run_belief_evolution_default_output(self):
        """Test plotting with default output directory."""

        # Create dummy data
        num_runs, num_rounds, num_agents = 3, 10, 16
        labels = np.random.randn(num_runs, num_rounds, num_agents)
        scores = np.random.randn(num_runs, num_rounds, num_agents)
        rounds = list(range(num_rounds))

        # Clean up any existing test output
        test_output = Path("outputs/plots/test_default_config_multi_run_evolution.png")
        if test_output.exists():
            test_output.unlink()

        try:
            # Call without output_dir (should use default outputs/plots)
            plot_multi_run_belief_evolution(
                labels, scores, rounds, "test_default_config"
            )

            # Check that plot was created in default location
            assert test_output.exists(), f"Plot not created at {test_output}"
        finally:
            # Clean up
            if test_output.exists():
                test_output.unlink()


class TestGenerateComparisonReport:
    """Test comparison report generation."""

    def test_generate_comparison_report(self):
        """Test generating comparison report."""
        # Find multi-run configurations
        multi_run_configs = find_multi_run_experiments("outputs/runs")

        if not multi_run_configs:
            pytest.skip("No multi-run experiments available")

        # Use first 2 configurations
        limited_configs = dict(list(multi_run_configs.items())[:2])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            df = generate_comparison_report(limited_configs, output_dir)

            # Should return a DataFrame
            import pandas as pd

            assert isinstance(df, pd.DataFrame)

            if not df.empty:
                # Check expected columns
                assert "configuration" in df.columns
                assert "num_runs" in df.columns
                assert "final_belief_mean" in df.columns

                # Check output file was created
                report_file = output_dir / "multi_run_comparison_report.csv"
                assert report_file.exists()

    def test_generate_comparison_report_empty(self):
        """Test generating comparison report with no data."""
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            df = generate_comparison_report({}, output_dir)

            assert isinstance(df, pd.DataFrame)
            assert df.empty
