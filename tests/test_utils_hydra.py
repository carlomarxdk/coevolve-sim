"""Tests for utils_hydra module.

Validates the build_catalog function to ensure it correctly constructs
agent catalogs from configuration specifications with proper id, name,
role, and prompt fields.
"""

import pytest

from src.utils import build_catalog


class TestBuildCatalogProcedural:
    """Tests for build_catalog in procedural mode."""

    def test_procedural_single_role_template(self):
        """Test procedural mode with a single role template."""
        spec = {
            "mode": "procedural",
            "role_templates": {
                "LLM": {
                    "name": "llama-base",
                    "role": "Human Participant",
                    "prompt": "You are a helpful assistant.",
                }
            },
            "counts": {"LLM": 3},
        }

        catalog = build_catalog(spec)

        assert len(catalog) == 3
        for i, agent in enumerate(catalog):
            assert agent["id"] == i
            assert agent["name"] == "llama-base"
            assert agent["role"] == "Human Participant"
            assert agent["prompt"] == "You are a helpful assistant."

    def test_procedural_multiple_role_templates(self):
        """Test procedural mode with multiple role templates."""
        spec = {
            "mode": "procedural",
            "role_templates": {
                "Physician": {
                    "name": "llama-doc",
                    "role": "Clinical Physician",
                    "prompt": "You are a clinical physician.",
                },
                "Engineer": {
                    "name": "llama-coder",
                    "role": "Software Engineer",
                    "prompt": "You are a software engineer.",
                },
            },
            "counts": {"Physician": 2, "Engineer": 3},
        }

        catalog = build_catalog(spec)

        assert len(catalog) == 5

        # First 2 should be physicians
        for i in range(2):
            assert catalog[i]["id"] == i
            assert catalog[i]["name"] == "llama-doc"
            assert catalog[i]["role"] == "Clinical Physician"
            assert catalog[i]["prompt"] == "You are a clinical physician."

        # Next 3 should be engineers
        for i in range(2, 5):
            assert catalog[i]["id"] == i
            assert catalog[i]["name"] == "llama-coder"
            assert catalog[i]["role"] == "Software Engineer"
            assert catalog[i]["prompt"] == "You are a software engineer."

    def test_procedural_empty_catalog(self):
        """Test procedural mode with no agents."""
        spec = {
            "mode": "procedural",
            "role_templates": {
                "LLM": {
                    "name": "llama-base",
                    "role": "Human Participant",
                    "prompt": "You are a helpful assistant.",
                }
            },
            "counts": {},
        }

        catalog = build_catalog(spec)

        assert len(catalog) == 0

    def test_procedural_unique_ids(self):
        """Test that IDs are unique and sequential in procedural mode."""
        spec = {
            "mode": "procedural",
            "role_templates": {
                "RoleA": {
                    "name": "model-a",
                    "role": "Role A",
                    "prompt": "Prompt A",
                },
                "RoleB": {
                    "name": "model-b",
                    "role": "Role B",
                    "prompt": "Prompt B",
                },
            },
            "counts": {"RoleA": 5, "RoleB": 3},
        }

        catalog = build_catalog(spec)

        ids = [agent["id"] for agent in catalog]
        assert ids == list(range(8))
        assert len(set(ids)) == 8


class TestBuildCatalogExplicit:
    """Tests for build_catalog in explicit mode."""

    def test_explicit_basic(self):
        """Test explicit mode with basic agent definitions."""
        spec = {
            "mode": "explicit",
            "explicit": [
                {
                    "id": 0,
                    "name": "llama-doc",
                    "role": "Clinical Physician",
                    "prompt": "You are a clinical physician.",
                },
                {
                    "id": 1,
                    "name": "llama-base",
                    "role": "Human Participant",
                    "prompt": "You are a participant.",
                },
            ],
        }

        catalog = build_catalog(spec)

        assert len(catalog) == 2
        assert catalog[0]["id"] == 0
        assert catalog[0]["name"] == "llama-doc"
        assert catalog[0]["role"] == "Clinical Physician"
        assert catalog[0]["prompt"] == "You are a clinical physician."

        assert catalog[1]["id"] == 1
        assert catalog[1]["name"] == "llama-base"
        assert catalog[1]["role"] == "Human Participant"
        assert catalog[1]["prompt"] == "You are a participant."

    def test_explicit_empty(self):
        """Test explicit mode with no agents."""
        spec = {"mode": "explicit", "explicit": []}

        catalog = build_catalog(spec)

        assert len(catalog) == 0

    def test_explicit_single_agent(self):
        """Test explicit mode with a single agent."""
        spec = {
            "mode": "explicit",
            "explicit": [
                {
                    "id": 42,
                    "name": "llama-special",
                    "role": "Special Agent",
                    "prompt": "You are special.",
                }
            ],
        }

        catalog = build_catalog(spec)

        assert len(catalog) == 1
        assert catalog[0]["id"] == 42
        assert catalog[0]["name"] == "llama-special"
        assert catalog[0]["role"] == "Special Agent"
        assert catalog[0]["prompt"] == "You are special."

    def test_explicit_preserves_all_fields(self):
        """Test that explicit mode preserves all provided agent fields."""
        spec = {
            "mode": "explicit",
            "explicit": [
                {
                    "id": 10,
                    "name": "custom-model",
                    "role": "Researcher",
                    "prompt": "Research prompt template",
                }
            ],
        }

        catalog = build_catalog(spec)

        agent = catalog[0]
        assert "id" in agent
        assert "name" in agent
        assert "role" in agent
        assert "prompt" in agent


class TestBuildCatalogRandomRoles:
    """Tests for build_catalog with random_roles feature."""

    def test_random_roles_deterministic_with_seed(self):
        """Test that random_roles produces deterministic results with seed."""
        spec = {
            "mode": "procedural",
            "role_templates": {
                "LLM": {
                    "name": "llama-base",
                    "role": "",
                    "prompt": "You are an agent.",
                }
            },
            "counts": {"LLM": 4},
            "random_roles": True,
            "random_roles_spec": {
                "Role A": 2,
                "Role B": 2,
            },
        }

        catalog1 = build_catalog(spec, seed=42)
        catalog2 = build_catalog(spec, seed=42)

        assert len(catalog1) == 4
        assert len(catalog2) == 4

        # Same seed should produce identical role assignments
        roles1 = [agent["role"] for agent in catalog1]
        roles2 = [agent["role"] for agent in catalog2]
        assert roles1 == roles2

    def test_random_roles_different_seeds_produce_different_results(self):
        """Test that different seeds produce different role assignments."""
        spec = {
            "mode": "procedural",
            "role_templates": {
                "LLM": {
                    "name": "llama-base",
                    "role": "",
                    "prompt": "You are an agent.",
                }
            },
            "counts": {"LLM": 4},
            "random_roles": True,
            "random_roles_spec": {
                "Role A": 2,
                "Role B": 2,
            },
        }

        catalog1 = build_catalog(spec, seed=42)
        catalog2 = build_catalog(spec, seed=99)

        roles1 = [agent["role"] for agent in catalog1]
        roles2 = [agent["role"] for agent in catalog2]

        # Different seeds should (with high probability) produce different assignments
        # We check that at least one role differs
        assert roles1 != roles2 or len(set(roles1)) == 1  # Allow edge case of all same

    def test_random_roles_assigns_correct_counts(self):
        """Test that random_roles assigns the correct number of each role."""
        spec = {
            "mode": "procedural",
            "role_templates": {
                "LLM": {
                    "name": "llama-base",
                    "role": "",
                    "prompt": "You are an agent.",
                }
            },
            "counts": {"LLM": 6},
            "random_roles": True,
            "random_roles_spec": {
                "Physician": 2,
                "Engineer": 3,
                "Scientist": 1,
            },
        }

        catalog = build_catalog(spec, seed=42)

        roles = [agent["role"] for agent in catalog]
        assert roles.count("Physician") == 2
        assert roles.count("Engineer") == 3
        assert roles.count("Scientist") == 1

    def test_random_roles_preserves_other_fields(self):
        """Test that random_roles only changes role field."""
        spec = {
            "mode": "procedural",
            "role_templates": {
                "LLM": {
                    "name": "llama-base",
                    "role": "Original Role",
                    "prompt": "Original prompt",
                }
            },
            "counts": {"LLM": 3},
            "random_roles": True,
            "random_roles_spec": {
                "New Role A": 1,
                "New Role B": 2,
            },
        }

        catalog = build_catalog(spec, seed=42)

        for i, agent in enumerate(catalog):
            assert agent["id"] == i
            assert agent["name"] == "llama-base"
            assert agent["prompt"] == "Original prompt"
            # Role should be from random_roles_spec
            assert agent["role"] in ["New Role A", "New Role B"]

    def test_random_roles_without_seed(self):
        """Test that random_roles works without explicit seed."""
        spec = {
            "mode": "procedural",
            "role_templates": {
                "LLM": {
                    "name": "llama-base",
                    "role": "",
                    "prompt": "You are an agent.",
                }
            },
            "counts": {"LLM": 4},
            "random_roles": True,
            "random_roles_spec": {
                "Role A": 2,
                "Role B": 2,
            },
        }

        catalog = build_catalog(spec)

        assert len(catalog) == 4
        roles = [agent["role"] for agent in catalog]
        assert roles.count("Role A") == 2
        assert roles.count("Role B") == 2


class TestBuildCatalogErrorHandling:
    """Tests for error handling in build_catalog."""

    def test_unknown_mode_raises_error(self):
        """Test that an unknown mode raises ValueError."""
        spec = {"mode": "unknown_mode"}

        with pytest.raises(ValueError, match="Unknown catalog mode"):
            build_catalog(spec)

    def test_random_roles_without_spec_raises_error(self):
        """Test that random_roles=True without random_roles_spec raises error."""
        spec = {
            "mode": "procedural",
            "role_templates": {
                "LLM": {
                    "name": "llama-base",
                    "role": "Human Participant",
                    "prompt": "You are an agent.",
                }
            },
            "counts": {"LLM": 3},
            "random_roles": True,
            "random_roles_spec": {},
        }

        with pytest.raises(
            ValueError, match="random_roles is enabled but random_roles_spec is empty"
        ):
            build_catalog(spec)

    def test_random_roles_count_mismatch_raises_error(self):
        """Test that mismatched counts in random_roles_spec raises error."""
        spec = {
            "mode": "procedural",
            "role_templates": {
                "LLM": {
                    "name": "llama-base",
                    "role": "",
                    "prompt": "You are an agent.",
                }
            },
            "counts": {"LLM": 5},
            "random_roles": True,
            "random_roles_spec": {
                "Role A": 2,
                "Role B": 2,
            },  # Total: 4, but catalog has 5
        }

        with pytest.raises(
            ValueError,
            match="Sum of random_roles_spec counts does not match number of agents",
        ):
            build_catalog(spec)


class TestBuildCatalogRequiredFields:
    """Tests to verify all agents have required fields."""

    @pytest.mark.parametrize(
        "spec",
        [
            # Procedural mode
            {
                "mode": "procedural",
                "role_templates": {
                    "LLM": {
                        "name": "llama-base",
                        "role": "Agent",
                        "prompt": "Prompt text",
                    }
                },
                "counts": {"LLM": 5},
            },
            # Explicit mode
            {
                "mode": "explicit",
                "explicit": [
                    {
                        "id": 0,
                        "name": "model-a",
                        "role": "Role A",
                        "prompt": "Prompt A",
                    },
                    {
                        "id": 1,
                        "name": "model-b",
                        "role": "Role B",
                        "prompt": "Prompt B",
                    },
                ],
            },
        ],
    )
    def test_all_agents_have_required_fields(self, spec):
        """Test that every agent has id, name, role, and prompt fields."""
        catalog = build_catalog(spec, seed=42)

        required_fields = ["id", "name", "role", "prompt"]

        for agent in catalog:
            for field in required_fields:
                assert field in agent, f"Agent missing required field: {field}"
                assert (
                    agent[field] is not None
                ), f"Agent field {field} should not be None"

    def test_procedural_with_random_roles_has_required_fields(self):
        """Test that procedural mode with random_roles has all required fields."""
        spec = {
            "mode": "procedural",
            "role_templates": {
                "LLM": {
                    "name": "llama-base",
                    "role": "",
                    "prompt": "You are an agent.",
                }
            },
            "counts": {"LLM": 4},
            "random_roles": True,
            "random_roles_spec": {
                "Role A": 2,
                "Role B": 2,
            },
        }

        catalog = build_catalog(spec, seed=42)

        required_fields = ["id", "name", "role", "prompt"]

        for agent in catalog:
            for field in required_fields:
                assert field in agent
                assert agent[field] is not None
                # Role should not be empty string after random assignment
                if field == "role":
                    assert agent[field] != ""


class TestBuildCatalogIntegration:
    """Integration tests using realistic catalog configurations."""

    def test_experts_catalog_structure(self):
        """Test a catalog similar to experts.yaml configuration."""
        spec = {
            "mode": "procedural",
            "role_templates": {
                "Clinical_Physician": {
                    "name": "llama-doc",
                    "role": "Clinical Physician",
                    "prompt": "physician_prompt",
                },
                "Software_Engineer": {
                    "name": "llama-coder",
                    "role": "Software Engineer",
                    "prompt": "engineer_prompt",
                },
                "Mathematician": {
                    "name": "llama-openmath",
                    "role": "Mathematician",
                    "prompt": "math_prompt",
                },
            },
            "counts": {
                "Clinical_Physician": 2,
                "Software_Engineer": 2,
                "Mathematician": 2,
            },
        }

        catalog = build_catalog(spec)

        assert len(catalog) == 6

        # Check that we have the right distribution
        roles = [agent["role"] for agent in catalog]
        assert roles.count("Clinical Physician") == 2
        assert roles.count("Software Engineer") == 2
        assert roles.count("Mathematician") == 2

        # Check sequential IDs
        ids = [agent["id"] for agent in catalog]
        assert ids == [0, 1, 2, 3, 4, 5]

    def test_random_roles_catalog_structure(self):
        """Test a catalog similar to random_roles.yaml configuration."""
        spec = {
            "mode": "procedural",
            "role_templates": {
                "LLM": {"name": "llama-base", "role": "", "prompt": "base_prompt"}
            },
            "counts": {"LLM": 8},
            "random_roles": True,
            "random_roles_spec": {
                "Assistant": 2,
                "Researcher": 2,
                "Engineer": 2,
                "Analyst": 2,
            },
        }

        catalog = build_catalog(spec, seed=123)

        assert len(catalog) == 8

        # Verify role distribution
        roles = [agent["role"] for agent in catalog]
        assert roles.count("Assistant") == 2
        assert roles.count("Researcher") == 2
        assert roles.count("Engineer") == 2
        assert roles.count("Analyst") == 2

        # All should have the same model name
        names = [agent["name"] for agent in catalog]
        assert all(name == "llama-base" for name in names)

        # IDs should be sequential
        ids = [agent["id"] for agent in catalog]
        assert ids == list(range(8))

    def test_explicit_catalog_structure(self):
        """Test a catalog similar to example_explicit.yaml configuration."""
        spec = {
            "mode": "explicit",
            "explicit": [
                {
                    "id": 0,
                    "name": "llama-doc",
                    "role": "Clinical Physician",
                    "prompt": "physician_template",
                },
                {
                    "id": 1,
                    "name": "llama-base",
                    "role": "Human Participant",
                    "prompt": "participant_template",
                },
                {
                    "id": 2,
                    "name": "llama-hermes",
                    "role": "Strategic Planner",
                    "prompt": "planner_template",
                },
            ],
        }

        catalog = build_catalog(spec)

        assert len(catalog) == 3

        # Verify each agent
        assert catalog[0]["id"] == 0
        assert catalog[0]["name"] == "llama-doc"
        assert catalog[0]["role"] == "Clinical Physician"

        assert catalog[1]["id"] == 1
        assert catalog[1]["name"] == "llama-base"
        assert catalog[1]["role"] == "Human Participant"

        assert catalog[2]["id"] == 2
        assert catalog[2]["name"] == "llama-hermes"
        assert catalog[2]["role"] == "Strategic Planner"
