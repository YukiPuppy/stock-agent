"""Read-only registry for LLM agent metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "name",
    "category",
    "level",
    "workflow",
    "default_model_env",
    "fallback_model_env",
    "frequency",
    "permission",
    "output_pattern",
    "description",
}
ALLOWED_PERMISSIONS = {"read_only", "proposal_only"}
REGISTRY_PATH = Path(__file__).resolve().parents[2] / "configs" / "agent_registry.json"


def list_agents() -> list[dict[str, Any]]:
    """Return all registered agents in config order."""
    return [_copy_agent(agent) for agent in _load_registry()]


def get_agent(name: str) -> dict[str, Any]:
    """Return one registered agent by name."""
    normalized_name = str(name).strip()
    for agent in _load_registry():
        if agent["name"] == normalized_name:
            return _copy_agent(agent)
    raise KeyError(f"Unknown agent: {name}")


def list_agents_by_category(category: str) -> list[dict[str, Any]]:
    """Return registered agents with the requested category."""
    normalized_category = str(category).strip()
    return [_copy_agent(agent) for agent in _load_registry() if agent["category"] == normalized_category]


def list_agents_by_workflow(workflow: str) -> list[dict[str, Any]]:
    """Return registered agents whose comma-separated workflow list includes workflow."""
    normalized_workflow = str(workflow).strip()
    return [
        _copy_agent(agent)
        for agent in _load_registry()
        if normalized_workflow in _split_csv_field(agent["workflow"])
    ]


def _load_registry() -> list[dict[str, Any]]:
    raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Agent registry must be a JSON list")

    seen_names: set[str] = set()
    agents: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Agent registry item {index} must be an object")
        missing_fields = REQUIRED_FIELDS.difference(item)
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"Agent registry item {index} missing field(s): {missing}")

        agent = {field: str(item[field]).strip() for field in REQUIRED_FIELDS}
        if not agent["name"]:
            raise ValueError(f"Agent registry item {index} has empty name")
        if agent["name"] in seen_names:
            raise ValueError(f"Duplicate agent registry name: {agent['name']}")
        if agent["permission"] not in ALLOWED_PERMISSIONS:
            raise ValueError(
                f"Agent {agent['name']} has unsupported permission: {agent['permission']}"
            )
        seen_names.add(agent["name"])
        agents.append(agent)
    return agents


def _split_csv_field(value: str) -> set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


def _copy_agent(agent: dict[str, Any]) -> dict[str, Any]:
    return dict(agent)
