"""
Tests for plot_role_vs_network.py script.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from plot_role_vs_network import (
    filter_by_roles,
    find_multi_run_experiments,
    load_role_network_data,
    plot_role_vs_network_comparison,
    plot_role_vs_network_single,
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


class TestLoadRoleNetworkData:
    """Test loading role and network data."""

    def test_load_role_network_data(self):
        """Test loading role network data from runs."""
        multi_run_configs = find_multi_run_experiments("outputs/runs")

        if not multi_run_configs:
            pytest.skip("No multi-run experiments available")

        config_name, run_dirs = next(iter(multi_run_configs.items()))

        df = load_role_network_data(run_dirs[:5])

        assert isinstance(df, pd.DataFrame)

        if not df.empty:
            expected_cols = [
                "run_id",
                "agent_id",
                "role",
                "degree",
                "eigenvector_centrality",
                "final_accuracy",
            ]
            for col in expected_cols:
                assert col in df.columns, f"Missing column: {col}"

            assert (df["final_accuracy"] >= 0).all()
            assert (df["final_accuracy"] <= 1).all()

    def test_load_role_network_data_empty_dirs(self):
        """Test loading role network data with empty run list."""
        df = load_role_network_data([])

        assert isinstance(df, pd.DataFrame)
        assert df.empty


class TestFilterByRoles:
    """Test role filtering."""

    def test_filter_by_roles_with_valid_roles(self):
        """Test filtering by specific roles."""
        df = pd.DataFrame(
            {
                "role": ["Chemist", "Physician", "Chemist", "Engineer"],
                "value": [1, 2, 3, 4],
            }
        )

        filtered = filter_by_roles(df, ["Chemist"])
        assert len(filtered) == 2
        assert all(filtered["role"] == "Chemist")

    def test_filter_by_roles_with_multiple_roles(self):
        """Test filtering by multiple roles."""
        df = pd.DataFrame(
            {
                "role": ["Chemist", "Physician", "Chemist", "Engineer"],
                "value": [1, 2, 3, 4],
            }
        )

        filtered = filter_by_roles(df, ["Chemist", "Physician"])
        assert len(filtered) == 3
        assert set(filtered["role"].unique()) == {"Chemist", "Physician"}

    def test_filter_by_roles_with_none(self):
        """Test that None roles returns original DataFrame."""
        df = pd.DataFrame({"role": ["Chemist", "Physician"], "value": [1, 2]})

        filtered = filter_by_roles(df, None)
        assert len(filtered) == len(df)

    def test_filter_by_roles_with_empty_list(self):
        """Test that empty list returns original DataFrame."""
        df = pd.DataFrame({"role": ["Chemist", "Physician"], "value": [1, 2]})

        filtered = filter_by_roles(df, [])
        assert len(filtered) == len(df)

    def test_filter_by_roles_nonexistent_role(self):
        """Test filtering by nonexistent role returns empty DataFrame."""
        df = pd.DataFrame({"role": ["Chemist", "Physician"], "value": [1, 2]})

        filtered = filter_by_roles(df, ["Nonexistent"])
        assert len(filtered) == 0


class TestPlotRoleVsNetworkSingle:
    """Test single experiment plotting."""

    def test_plot_role_vs_network_single(self):
        """Test generating a single experiment plot."""
        multi_run_configs = find_multi_run_experiments("outputs/runs")

        if not multi_run_configs:
            pytest.skip("No multi-run experiments available")

        config_name, run_dirs = next(iter(multi_run_configs.items()))
        df = load_role_network_data(run_dirs[:5])

        if df.empty:
            pytest.skip("No role/network data available")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            plot_role_vs_network_single(df, "test_experiment", output_dir)

            plots = list(output_dir.glob("*.pdf"))
            assert len(plots) == 1

    def test_plot_role_vs_network_single_empty_df(self):
        """Test plotting with empty DataFrame does not raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            df = pd.DataFrame()

            plot_role_vs_network_single(df, "empty_experiment", output_dir)

            plots = list(output_dir.glob("*.pdf"))
            assert len(plots) == 0


class TestPlotRoleVsNetworkComparison:
    """Test comparison plotting across experiments."""

    def test_plot_role_vs_network_comparison(self):
        """Test generating comparison plot across experiments."""
        multi_run_configs = find_multi_run_experiments("outputs/runs")

        if not multi_run_configs:
            pytest.skip("No multi-run experiments available")

        experiment_dfs = {}
        for config_name, run_dirs in multi_run_configs.items():
            parts = config_name.split("/")
            if len(parts) != 4:
                continue
            exp_name = parts[1]
            if exp_name in experiment_dfs:
                continue
            df = load_role_network_data(run_dirs[:5])
            if not df.empty:
                experiment_dfs[exp_name] = df
            if len(experiment_dfs) >= 2:
                break

        if len(experiment_dfs) < 2:
            pytest.skip("Not enough experiments for comparison")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            plot_role_vs_network_comparison(experiment_dfs, output_dir)

            plots = list(output_dir.glob("*.pdf"))
            assert len(plots) == 1

    def test_plot_role_vs_network_comparison_empty(self):
        """Test comparison with empty dict does not raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            plot_role_vs_network_comparison({}, output_dir)

            plots = list(output_dir.glob("*.pdf"))
            assert len(plots) == 0
