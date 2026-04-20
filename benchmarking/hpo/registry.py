"""HPO registry (lazy-populated)."""
from __future__ import annotations

from typing import Dict, Iterable, List

from benchmarking.hpo.contracts import HPOEntry

_REGISTRY: Dict[str, HPOEntry] = {}


def _ensure_registered() -> None:
    if _REGISTRY:
        return
    from benchmarking.hpo.register_entries import register_default_hpo_entries

    register_default_hpo_entries()


def register(entry: HPOEntry) -> None:
    _REGISTRY[entry.name] = entry


def get(model_name: str) -> HPOEntry:
    _ensure_registered()
    if model_name not in _REGISTRY:
        raise KeyError(f"No HPO registry entry for model: {model_name!r}")
    return _REGISTRY[model_name]


def list_registered() -> List[str]:
    _ensure_registered()
    return sorted(_REGISTRY.keys())


def iter_entries() -> Iterable[HPOEntry]:
    """Yield all registered entries (sorted by name)."""
    _ensure_registered()
    for name in sorted(_REGISTRY.keys()):
        yield _REGISTRY[name]


def clear_for_tests() -> None:
    _REGISTRY.clear()
