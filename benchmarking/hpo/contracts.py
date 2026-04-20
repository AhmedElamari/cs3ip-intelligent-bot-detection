"""HPO result schema and registry entry."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

if sys.version_info >= (3, 11):
    from typing import NotRequired, TypedDict
else:
    from typing_extensions import NotRequired, TypedDict


class HPOResultV1(TypedDict, total=False):
    schema_version: Literal["HPOResultV1"]
    status: Literal["ok", "failed", "skipped"]
    best_params: dict[str, Any]
    best_score: float
    trial_count: int
    metric: str
    seed: int
    warnings: list[str]
    model_name: str
    search_space_version: str
    device: NotRequired[str]


SuggestFn = Callable[[Any], dict[str, Any]]
ScoreFn = Callable[..., float]
PrunerFactory = Callable[[], Any]


@dataclass(frozen=True)
class HPOEntry:
    name: str
    search_space_version: str
    suggest_fn: SuggestFn
    score_fn: ScoreFn
    pruner_factory: Optional[PrunerFactory] = None
    requires_dl: bool = False
