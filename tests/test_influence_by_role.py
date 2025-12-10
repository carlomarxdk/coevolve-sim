"""Tests for influence scores by role analysis functionality."""

import tempfile
import unittest
from pathlib import Path

# Configure matplotlib backend before importing pyplot
import matplotlib
import pandas as pd

matplotlib.use("Agg")  # Use non-interactive backend for testing

from plot_statistics import (
    analyze_influence_scores_by_role,
    plot_influence_scores_by_role,
)


class TestInfluenceByRole(unittest.TestCase):
    """Test influence scores by role analysis and plotting."""

    def setUp(self):
        """Set up test data."""
        self.sample_data = {
            "agent_manifest": {
                "0": {"agent_id": 0, "role": "Human Participant"},
                "1": {"agent_id": 1, "role": "Chemist"},
                "2": {"agent_id": 2, "role": "Human Participant"},
                "3": {"agent_id": 3, "role": "Storyteller"},
                "4": {"agent_id": 4, "role": "Chemist"},
                "5": {"agent_id": 5, "role": "Human Participant"},
            },
            "final_metrics": {
                "influence_scores_granger": [0.5, 0.8, 0.3, 0.9, 0.7, 0.4],
                "influence_scores_simple": [0.6, 0.7, 0.5, 0.95, 0.85, 0.45],
                "information_leadership": [0.55, 0.75, 0.35, 0.92, 0.78, 0.42],
            },
        }

    def test_analyze_influence_scores_by_role(self):
        """Test analysis of influence scores by role."""
        df = analyze_influence_scores_by_role(self.sample_data)

        # Check DataFrame structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIn("role", df.columns)
        self.assertIn("agent_id", df.columns)
        self.assertIn("influence_granger", df.columns)
        self.assertIn("influence_simple", df.columns)
        self.assertIn("information_leadership", df.columns)

        # Check number of rows matches number of agents
        self.assertEqual(len(df), 6)

        # Check role assignments
        roles = df["role"].unique()
        expected_roles = {"Human Participant", "Chemist", "Storyteller"}
        self.assertEqual(set(roles), expected_roles)

        # Check specific values
        agent_0_data = df[df["agent_id"] == 0].iloc[0]
        self.assertEqual(agent_0_data["role"], "Human Participant")
        self.assertEqual(agent_0_data["influence_granger"], 0.5)
        self.assertEqual(agent_0_data["influence_simple"], 0.6)
        self.assertEqual(agent_0_data["information_leadership"], 0.55)

    def test_analyze_influence_scores_empty_data(self):
        """Test analysis with missing data."""
        empty_data = {}
        df = analyze_influence_scores_by_role(empty_data)
        self.assertTrue(df.empty)

        no_metrics_data = {
            "agent_manifest": self.sample_data["agent_manifest"],
            "final_metrics": {},
        }
        df = analyze_influence_scores_by_role(no_metrics_data)
        self.assertTrue(df.empty)

    def test_analyze_influence_scores_partial_metrics(self):
        """Test analysis with only some influence metrics available."""
        partial_data = {
            "agent_manifest": {
                "0": {"agent_id": 0, "role": "Human Participant"},
                "1": {"agent_id": 1, "role": "Chemist"},
            },
            "final_metrics": {
                "influence_scores_simple": [0.6, 0.7],
            },
        }
        df = analyze_influence_scores_by_role(partial_data)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertIn("influence_simple", df.columns)
        self.assertNotIn("influence_granger", df.columns)

    def test_role_aggregation(self):
        """Test that influence scores can be aggregated by role."""
        df = analyze_influence_scores_by_role(self.sample_data)

        # Aggregate by role (not currently validated)
        # role_stats = df.groupby("role").agg(
        #     {
        #         "influence_granger": ["mean", "std", "count"],
        #         "influence_simple": ["mean", "std", "count"],
        #         "information_leadership": ["mean", "std", "count"],
        #     }
        # )

        # Check Human Participant role
        hp_granger_mean = df[df["role"] == "Human Participant"][
            "influence_granger"
        ].mean()
        expected_hp_granger = (0.5 + 0.3 + 0.4) / 3
        self.assertAlmostEqual(hp_granger_mean, expected_hp_granger, places=6)

        # Check Chemist role
        chemist_simple_mean = df[df["role"] == "Chemist"]["influence_simple"].mean()
        expected_chemist_simple = (0.7 + 0.85) / 2
        self.assertAlmostEqual(chemist_simple_mean, expected_chemist_simple, places=6)

    def test_plot_influence_scores_by_role(self):
        """Test plotting of influence scores by role."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_influence_plot.png"

            # Should not raise an exception
            plot_influence_scores_by_role(self.sample_data, str(output_path))

            # Check that file was created
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_plot_influence_scores_empty_data(self):
        """Test plotting with empty data."""
        empty_data = {}

        # Should not raise an exception, just print a message
        plot_influence_scores_by_role(empty_data)

    def test_plot_influence_scores_no_output_path(self):
        """Test plotting without saving (just display)."""
        # Should not raise an exception
        plot_influence_scores_by_role(self.sample_data)


class TestInfluenceByRoleEdgeCases(unittest.TestCase):
    """Test edge cases for influence by role analysis."""

    def test_single_role(self):
        """Test with all agents having the same role."""
        data = {
            "agent_manifest": {
                "0": {"agent_id": 0, "role": "Human Participant"},
                "1": {"agent_id": 1, "role": "Human Participant"},
                "2": {"agent_id": 2, "role": "Human Participant"},
            },
            "final_metrics": {
                "influence_scores_granger": [0.5, 0.6, 0.7],
            },
        }
        df = analyze_influence_scores_by_role(data)

        self.assertEqual(len(df), 3)
        self.assertEqual(len(df["role"].unique()), 1)
        self.assertEqual(df["role"].unique()[0], "Human Participant")

    def test_zero_influence_scores(self):
        """Test with all zero influence scores (converged experiment)."""
        data = {
            "agent_manifest": {
                "0": {"agent_id": 0, "role": "Human Participant"},
                "1": {"agent_id": 1, "role": "Chemist"},
            },
            "final_metrics": {
                "influence_scores_granger": [0.0, 0.0],
                "influence_scores_simple": [0.0, 0.0],
                "information_leadership": [0.0, 0.0],
            },
        }
        df = analyze_influence_scores_by_role(data)

        self.assertEqual(len(df), 2)
        self.assertTrue((df["influence_granger"] == 0.0).all())
        self.assertTrue((df["influence_simple"] == 0.0).all())
        self.assertTrue((df["information_leadership"] == 0.0).all())

    def test_missing_role_field(self):
        """Test with agents missing role field."""
        data = {
            "agent_manifest": {
                "0": {"agent_id": 0},  # No role field
                "1": {"agent_id": 1, "role": "Chemist"},
            },
            "final_metrics": {
                "influence_scores_simple": [0.5, 0.6],
            },
        }
        df = analyze_influence_scores_by_role(data)

        self.assertEqual(len(df), 2)
        # Agent without role should get 'unknown'
        self.assertIn("unknown", df["role"].values)


if __name__ == "__main__":
    unittest.main()
