"""
Tests for IOManager functionality including experiment naming,
directory organization, and database tracking.
"""

import json
import pathlib

# Add parent directory to path
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from utils import (
    IOManager,
    check_experiment_completed,
    get_experiment_choices,
    move_incomplete_experiments,
)

# def test_io_manager_basic_initialization():
#     """Test basic IOManager initialization without experiment config."""
#     with tempfile.TemporaryDirectory() as tmpdir:
#         # Change to temp directory for this test
#         original_cwd = pathlib.Path.cwd()
#         try:
#             import os
#             os.chdir(tmpdir)

#             cfg = {'save_activations': True, 'save_text': True}
#             io = IOManager(cfg)

#             assert io.out_dir.exists()
#             assert 'experiments' in str(io.out_dir)
#             assert io.save_activations is True
#             assert io.save_text is True
#         finally:
#             os.chdir(original_cwd)


def test_io_manager_with_experiment_config():
    """Test IOManager with full experiment configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = pathlib.Path.cwd()
        try:
            import os

            os.chdir(tmpdir)

            experiment_cfg = {
                "statement": {"id": "t1", "statement": "Test statement"},
                "prompt": {"type": "wR_L"},
                "seed": 42,
                "experiment": {"max_rounds": 10},
                "network": {"generator": "erdos-renyi", "params": {"n": 5, "p": 0.5}},
            }

            cfg = {"save_activations": False, "save_text": True}
            io = IOManager(cfg, experiment_cfg=experiment_cfg)

            # Check that directory structure follows pattern: catalog/prompt/statement/timestamp
            # Without hydra metadata, it should use fallback to infer from config
            # path_parts = io.out_dir.parts  # Not currently validated

            # Should contain statement choice
            assert "t1" in str(io.out_dir)

            # Should contain prompt choice
            assert "wR_L" in str(io.out_dir)

            # Check config file was saved
            config_path = io.out_dir / "config.json"
            assert config_path.exists()

            with open(config_path, "r") as f:
                saved_config = json.load(f)
                assert saved_config["statement"]["id"] == "t1"
                assert saved_config["seed"] == 42
        finally:
            os.chdir(original_cwd)


def test_rounds_directory_structure():
    """Test that rounds are saved in organized subdirectory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = pathlib.Path.cwd()
        try:
            import os

            os.chdir(tmpdir)

            cfg = {}
            io = IOManager(cfg)

            # Test _round_path creates proper structure
            round_path = io._round_path(0)

            assert "rounds" in str(round_path)
            assert "round_0" in str(round_path)
            assert round_path.exists()
        finally:
            os.chdir(original_cwd)


def test_save_json_in_experiment_dir():
    """Test saving JSON files within experiment directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = pathlib.Path.cwd()
        try:
            import os

            os.chdir(tmpdir)

            cfg = {}
            io = IOManager(cfg)

            test_data = {"key": "value", "number": 42}
            saved_path = io.save_json("test/nested/file.json", test_data)

            assert saved_path.exists()
            assert saved_path.parent.name == "nested"

            with open(saved_path, "r") as f:
                loaded = json.load(f)
                assert loaded == test_data
        finally:
            os.chdir(original_cwd)


def test_hierarchical_directory_structure_with_hydra():
    """Test that IOManager creates correct hierarchical structure with Hydra metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = pathlib.Path.cwd()
        try:
            import os

            os.chdir(tmpdir)

            # Simulate config with Hydra runtime choices
            experiment_cfg = {
                "hydra": {
                    "runtime": {
                        "choices": {
                            "catalog": "only_llms",
                            "prompt": "wR_L",
                            "statement": "t1",
                        }
                    }
                },
                "statement": {"id": "t1", "statement": "Test statement"},
                "prompt": {"type": "wR_L"},
                "seed": 42,
            }

            cfg = {}
            io = IOManager(cfg, experiment_cfg=experiment_cfg)

            # Verify hierarchical structure: catalog/prompt/statement/timestamp
            path_str = str(io.out_dir)
            assert "only_llms" in path_str
            assert "wR_L" in path_str
            assert "t1" in path_str

            # Verify the order is correct
            parts = io.out_dir.parts
            # Find the indices
            catalog_idx = next(i for i, p in enumerate(parts) if p == "only_llms")
            prompt_idx = next(i for i, p in enumerate(parts) if p == "wR_L")
            statement_idx = next(i for i, p in enumerate(parts) if p == "t1")

            # Verify catalog comes before prompt, prompt before statement
            assert catalog_idx < prompt_idx < statement_idx

            # Check that config was saved with hydra choices preserved
            config_path = io.out_dir / "config.json"
            assert config_path.exists()

            with open(config_path, "r") as f:
                saved_config = json.load(f)
                # Verify hydra choices are preserved
                if "hydra" in saved_config:
                    assert (
                        saved_config["hydra"]["runtime"]["choices"]["catalog"]
                        == "only_llms"
                    )
                    assert (
                        saved_config["hydra"]["runtime"]["choices"]["prompt"] == "wR_L"
                    )
                    assert (
                        saved_config["hydra"]["runtime"]["choices"]["statement"] == "t1"
                    )
        finally:
            os.chdir(original_cwd)


def test_check_experiment_completed():
    """Test checking if an experiment with same seed and max_rounds is completed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = pathlib.Path.cwd()
        try:
            import os

            os.chdir(tmpdir)

            # Create a completed experiment structure
            base_dir = pathlib.Path(tmpdir) / "outputs" / "runs"
            catalog_choice = "only_llms"
            prompt_choice = "wR_L"
            statement_choice = "t1"
            seed = 42
            max_rounds = 10

            # Create the directory structure
            experiment_dir = (
                base_dir
                / catalog_choice
                / prompt_choice
                / statement_choice
                / "2025-01-01_12-00-00"
            )
            experiment_dir.mkdir(parents=True, exist_ok=True)

            # Save config with seed
            config_path = experiment_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump({"seed": seed}, f)

            # Create rounds directory with completed rounds (0 to 9 for max_rounds=10)
            rounds_dir = experiment_dir / "rounds"
            for i in range(max_rounds):
                round_dir = rounds_dir / f"round_{i}"
                round_dir.mkdir(parents=True, exist_ok=True)
                beliefs_path = round_dir / "beliefs.jsonl"
                with open(beliefs_path, "w") as f:
                    f.write('{"round": ' + str(i) + ', "belief": 1}\n')

            # Test that the experiment is detected as completed
            result = check_experiment_completed(
                catalog_choice,
                prompt_choice,
                statement_choice,
                seed,
                max_rounds,
                str(base_dir),
            )
            assert result is True, "Should detect completed experiment"

            # Test with different seed - should not be completed
            result = check_experiment_completed(
                catalog_choice,
                prompt_choice,
                statement_choice,
                999,
                max_rounds,
                str(base_dir),
            )
            assert result is False, "Should not detect experiment with different seed"

            # Test with incomplete rounds - create new experiment
            incomplete_experiment_dir = (
                base_dir
                / catalog_choice
                / prompt_choice
                / statement_choice
                / "2025-01-01_12-30-00"
            )
            incomplete_experiment_dir.mkdir(parents=True, exist_ok=True)

            # Save config with same seed
            incomplete_config_path = incomplete_experiment_dir / "config.json"
            with open(incomplete_config_path, "w") as f:
                json.dump({"seed": seed}, f)

            # Create only partial rounds (0 to 5)
            incomplete_rounds_dir = incomplete_experiment_dir / "rounds"
            for i in range(6):
                round_dir = incomplete_rounds_dir / f"round_{i}"
                round_dir.mkdir(parents=True, exist_ok=True)
                beliefs_path = round_dir / "beliefs.jsonl"
                with open(beliefs_path, "w") as f:
                    f.write('{"round": ' + str(i) + ', "belief": 1}\n')

            # Should still return True because first experiment is complete
            result = check_experiment_completed(
                catalog_choice,
                prompt_choice,
                statement_choice,
                seed,
                max_rounds,
                str(base_dir),
            )
            assert (
                result is True
            ), "Should detect completed experiment even with incomplete one present"

            # Test with missing middle round
            missing_middle_experiment_dir = (
                base_dir
                / catalog_choice
                / prompt_choice
                / statement_choice
                / "2025-01-01_14-00-00"
            )
            missing_middle_experiment_dir.mkdir(parents=True, exist_ok=True)

            # Save config with same seed
            missing_middle_config_path = missing_middle_experiment_dir / "config.json"
            with open(missing_middle_config_path, "w") as f:
                json.dump({"seed": seed}, f)

            # Create all rounds except round 5 (missing middle round)
            missing_middle_rounds_dir = missing_middle_experiment_dir / "rounds"
            for i in range(max_rounds):
                if i == 5:  # Skip round 5
                    continue
                round_dir = missing_middle_rounds_dir / f"round_{i}"
                round_dir.mkdir(parents=True, exist_ok=True)
                beliefs_path = round_dir / "beliefs.jsonl"
                with open(beliefs_path, "w") as f:
                    f.write('{"round": ' + str(i) + ', "belief": 1}\n')

            # Should still return True because the first complete experiment exists
            result = check_experiment_completed(
                catalog_choice,
                prompt_choice,
                statement_choice,
                seed,
                max_rounds,
                str(base_dir),
            )
            assert (
                result is True
            ), "Should detect completed experiment even when another has missing rounds"

        finally:
            os.chdir(original_cwd)


def test_check_experiment_with_missing_middle_round():
    """Test that experiments with missing middle rounds are not marked as completed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = pathlib.Path.cwd()
        try:
            import os

            os.chdir(tmpdir)

            base_dir = pathlib.Path(tmpdir) / "outputs" / "runs"
            catalog_choice = "test_catalog"
            prompt_choice = "test_prompt"
            statement_choice = "test_statement"
            seed = 77777
            max_rounds = 10

            # Create experiment with missing middle round
            experiment_dir = (
                base_dir
                / catalog_choice
                / prompt_choice
                / statement_choice
                / "2025-01-01_15-00-00"
            )
            experiment_dir.mkdir(parents=True, exist_ok=True)

            # Save config with seed
            config_path = experiment_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump({"seed": seed}, f)

            # Create all rounds except round 5 (missing middle round)
            rounds_dir = experiment_dir / "rounds"
            for i in range(max_rounds):
                if i == 5:  # Skip round 5
                    continue
                round_dir = rounds_dir / f"round_{i}"
                round_dir.mkdir(parents=True, exist_ok=True)
                beliefs_path = round_dir / "beliefs.jsonl"
                with open(beliefs_path, "w") as f:
                    f.write('{"round": ' + str(i) + ', "belief": 1}\n')

            # Should NOT be detected as completed due to missing round
            result = check_experiment_completed(
                catalog_choice,
                prompt_choice,
                statement_choice,
                seed,
                max_rounds,
                str(base_dir),
            )
            assert (
                result is False
            ), "Should not detect experiment with missing middle round as completed"

        finally:
            os.chdir(original_cwd)


def test_get_experiment_choices():
    """Test extracting experiment choices from config."""
    # Test with hydra runtime choices
    experiment_cfg = {
        "hydra": {
            "runtime": {
                "choices": {"catalog": "only_llms", "prompt": "wR_L", "statement": "t1"}
            }
        }
    }

    catalog, prompt, statement = get_experiment_choices(experiment_cfg)
    assert catalog == "only_llms"
    assert prompt == "wR_L"
    assert statement == "t1"

    # Test with fallback to config structure
    experiment_cfg = {
        "statement": {"id": "t2"},
        "prompt": {"type": "woR_C"},
        "agents": {"catalog": [{"role": "LLM"}, {"role": "LLM"}]},
    }

    catalog, prompt, statement = get_experiment_choices(experiment_cfg)
    assert catalog == "only_llms"
    assert prompt == "woR_C"
    assert statement == "t2"


def test_move_incomplete_experiments():
    """Test moving incomplete experiments to incomplete_runs directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = pathlib.Path.cwd()
        try:
            import os

            os.chdir(tmpdir)

            base_dir = pathlib.Path(tmpdir) / "outputs" / "runs"
            incomplete_dir = pathlib.Path(tmpdir) / "outputs" / "incomplete_runs"
            catalog_choice = "test_catalog"
            prompt_choice = "test_prompt"
            statement_choice = "test_statement"
            seed = 12345
            max_rounds = 10

            # Create an incomplete experiment
            experiment_dir = (
                base_dir
                / catalog_choice
                / prompt_choice
                / statement_choice
                / "2025-01-01_10-00-00"
            )
            experiment_dir.mkdir(parents=True, exist_ok=True)

            # Save config with seed
            config_path = experiment_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump({"seed": seed}, f)

            # Create only 5 rounds (incomplete)
            rounds_dir = experiment_dir / "rounds"
            for i in range(5):
                round_dir = rounds_dir / f"round_{i}"
                round_dir.mkdir(parents=True, exist_ok=True)
                beliefs_path = round_dir / "beliefs.jsonl"
                with open(beliefs_path, "w") as f:
                    f.write('{"round": ' + str(i) + ', "belief": 1, "agent": 0}\n')

            # Move incomplete experiments
            moved_count = move_incomplete_experiments(
                catalog_choice,
                prompt_choice,
                statement_choice,
                seed,
                max_rounds,
                str(base_dir),
                str(incomplete_dir),
            )

            assert moved_count == 1, "Should move 1 incomplete experiment"

            # Check that the experiment was moved
            assert not experiment_dir.exists(), "Original directory should not exist"

            moved_dir = (
                incomplete_dir
                / catalog_choice
                / prompt_choice
                / statement_choice
                / "2025-01-01_10-00-00"
            )
            assert moved_dir.exists(), "Experiment should be in incomplete_runs"

            # Verify the moved experiment still has its data
            moved_config = moved_dir / "config.json"
            assert moved_config.exists(), "Config should exist in moved location"

            # Verify no incomplete experiments remain in original location
            result = check_experiment_completed(
                catalog_choice,
                prompt_choice,
                statement_choice,
                seed,
                max_rounds,
                str(base_dir),
            )
            assert (
                result is False
            ), "Should not find completed experiment after moving incomplete"

        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    # Run tests
    import pytest

    pytest.main([__file__, "-v"])
