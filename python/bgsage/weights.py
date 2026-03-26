"""Weight configuration for the 5-NN game plan strategy.

The PRODUCTION_MODEL constant defines which model all scripts and the analyzer
use by default.  Change it here — and only here — to promote a new model.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Each entry maps a short name to (hidden_sizes, weight_file_pattern).
# weight_file_pattern uses {plan} as placeholder for the plan name
# (purerace, racing, attacking, priming, anchoring).

MODELS: dict[str, dict[str, Any]] = {
    "stage8": {
        "hidden": (100,) + (400,) * 16,
        "pattern": "sl_s8_{plan}.weights.best",
        "plans": "pair",  # uses _PAIR_PLANS (17 NNs)
        # NN sharing: same as S7 — indices 11, 15, 16 share index 12.
        "canonical_map": [0,1,2,3,4,5,6,7,8,9,10,12,12,13,14,12,12],
    },
    "stage7": {
        "hidden": (100,) + (300,) * 16,
        "pattern": "sl_s7_{plan}.weights.best",
        "plans": "pair",  # uses _PAIR_PLANS (17 NNs)
        # NN sharing: indices 11 (prim_prim), 15 (anch_prim), 16 (anch_anch)
        # share the NN at index 12 (prim_anch).
        "canonical_map": [0,1,2,3,4,5,6,7,8,9,10,12,12,13,14,12,12],
    },
    "stage6": {
        "hidden": (100, 300, 300, 300, 300),
        "pattern": "sl_s6_{plan}.weights.best",
    },
    "stage5": {
        "hidden": (200, 400, 400, 400, 400),
        "pattern": "sl_s5_{plan}.weights.best",
    },
    "stage5small": {
        "hidden": (100, 200, 200, 200, 200),
        "pattern": "sl_s5s_{plan}.weights.best",
    },
    "stage4": {
        "hidden": (120, 250, 250, 250, 250),
        "pattern": "sl_s4_{plan}.weights.best",
    },
    "stage3": {
        "hidden": (120, 250, 250, 250, 250),
        "pattern": "sl_{plan}.weights.best",
    },
}

# *** THE production model — change this single line to promote a new model ***
PRODUCTION_MODEL: str = "stage8"

_PLANS = ("purerace", "racing", "attacking", "priming", "anchoring")

# 17-NN pair plan names (1 purerace + 16 ordered contact pairs)
_PAIR_PLANS = (
    "purerace",
    "race_race", "race_att", "race_prim", "race_anch",
    "att_race", "att_att", "att_prim", "att_anch",
    "prim_race", "prim_att", "prim_prim", "prim_anch",
    "anch_race", "anch_att", "anch_prim", "anch_anch",
)

# Bearoff database filename (stored in data/ directory)
BEAROFF_DB_FILENAME = "bearoff_1sided.db"


def _models_dir() -> str:
    """Return the absolute path to bgsage/models/."""
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "models")
    )


def _data_dir() -> str:
    """Return the absolute path to bgsage/data/."""
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "data")
    )


def bearoff_db_path() -> str | None:
    """Return the path to the bearoff database file, or None if not found."""
    path = os.path.join(_data_dir(), BEAROFF_DB_FILENAME)
    return path if os.path.exists(path) else None


@dataclass
class WeightConfig:
    """Paths to the 5-NN weight files and their hidden-layer sizes."""

    purerace: str
    racing: str
    attacking: str
    priming: str
    anchoring: str
    n_hidden_purerace: int = 200
    n_hidden_racing: int = 400
    n_hidden_attacking: int = 400
    n_hidden_priming: int = 400
    n_hidden_anchoring: int = 400

    @property
    def weight_args(self) -> tuple:
        """10-tuple for passing to C++ factory functions."""
        return (
            self.purerace,
            self.racing,
            self.attacking,
            self.priming,
            self.anchoring,
            self.n_hidden_purerace,
            self.n_hidden_racing,
            self.n_hidden_attacking,
            self.n_hidden_priming,
            self.n_hidden_anchoring,
        )

    @property
    def hidden_sizes(self) -> tuple[int, int, int, int, int]:
        """(purerace, racing, attacking, priming, anchoring) hidden sizes."""
        return (
            self.n_hidden_purerace,
            self.n_hidden_racing,
            self.n_hidden_attacking,
            self.n_hidden_priming,
            self.n_hidden_anchoring,
        )

    @property
    def weight_paths(self) -> dict[str, str]:
        """Dict mapping plan name to weight file path."""
        return {
            "purerace": self.purerace,
            "racing": self.racing,
            "attacking": self.attacking,
            "priming": self.priming,
            "anchoring": self.anchoring,
        }

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def default(cls) -> WeightConfig:
        """Return the production model's WeightConfig.

        Uses the PRODUCTION_MODEL constant defined at the top of this module.
        """
        return cls.from_model(PRODUCTION_MODEL)

    @classmethod
    def from_model(cls, name: str) -> WeightConfig:
        """Build a WeightConfig from a registered model name.

        Raises KeyError if the name is not in MODELS.
        """
        if name not in MODELS:
            available = ", ".join(sorted(MODELS))
            raise KeyError(
                f"Unknown model {name!r}. Available: {available}"
            )
        cfg = MODELS[name]
        hidden = cfg["hidden"]
        pattern = cfg["pattern"]
        models_dir = _models_dir()
        paths = {
            plan: os.path.join(models_dir, pattern.format(plan=plan))
            for plan in _PLANS
        }
        return cls(
            purerace=paths["purerace"],
            racing=paths["racing"],
            attacking=paths["attacking"],
            priming=paths["priming"],
            anchoring=paths["anchoring"],
            n_hidden_purerace=hidden[0],
            n_hidden_racing=hidden[1],
            n_hidden_attacking=hidden[2],
            n_hidden_priming=hidden[3],
            n_hidden_anchoring=hidden[4],
        )

    # ------------------------------------------------------------------
    # argparse helpers for scripts
    # ------------------------------------------------------------------

    @staticmethod
    def add_model_arg(parser: argparse.ArgumentParser) -> None:
        """Add a ``--model`` argument that defaults to the production model."""
        parser.add_argument(
            "--model",
            type=str,
            default=PRODUCTION_MODEL,
            help=(
                f"Model name from the registry (default: {PRODUCTION_MODEL}). "
                f"Available: {', '.join(sorted(MODELS))}"
            ),
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> WeightConfig:
        """Build a WeightConfig from parsed argparse args (expects ``args.model``)."""
        return cls.from_model(args.model)

    def validate(self) -> None:
        """Check that all weight files exist. Raises FileNotFoundError if not."""
        for plan, path in self.weight_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"{plan} weights not found: {path}")

    def print_summary(self, label: str | None = None) -> None:
        """Print a short summary of the model configuration."""
        if label:
            print(f"=== {label} ===")
        for plan in _PLANS:
            path = self.weight_paths[plan]
            h = getattr(self, f"n_hidden_{plan}")
            print(f"  {plan:10s}: {path} ({h}h)")
        print()


@dataclass
class WeightConfigPair:
    """Paths and hidden sizes for 17-NN pair strategy models.

    Stores weight paths and hidden sizes as lists (index 0 = purerace,
    indices 1-16 = ordered contact pairs). Use with GamePlanPairStrategy.
    """

    paths: list[str]         # 17 weight file paths
    hiddens: list[int]       # 17 hidden sizes

    @property
    def weight_args(self) -> tuple[list[str], list[int]]:
        """(weight_paths, hidden_sizes) for passing to C++ pair bindings."""
        return (self.paths, self.hiddens)

    @property
    def weight_paths(self) -> dict[str, str]:
        """Dict mapping pair name to weight file path."""
        return {name: path for name, path in zip(_PAIR_PLANS, self.paths)}

    @classmethod
    def from_model(cls, name: str) -> WeightConfigPair:
        """Build from a registered pair model name.

        Handles NN sharing via canonical_map: aliased indices use the
        canonical plan's weight file path.
        """
        if name not in MODELS:
            raise KeyError(f"Unknown model {name!r}")
        cfg = MODELS[name]
        if cfg.get("plans") != "pair":
            raise ValueError(f"Model {name!r} is not a pair model")
        hidden = list(cfg["hidden"])
        pattern = cfg["pattern"]
        models_dir = _models_dir()
        canonical_map = cfg.get("canonical_map", list(range(len(_PAIR_PLANS))))
        paths = []
        for i, plan in enumerate(_PAIR_PLANS):
            canonical_plan = _PAIR_PLANS[canonical_map[i]]
            paths.append(os.path.join(models_dir, pattern.format(plan=canonical_plan)))
        return cls(paths=paths, hiddens=hidden)

    def validate(self) -> None:
        """Check that all weight files exist."""
        for name, path in zip(_PAIR_PLANS, self.paths):
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} weights not found: {path}")

    def print_summary(self, label: str | None = None) -> None:
        """Print a short summary of the model configuration."""
        if label:
            print(f"=== {label} ===")
        for name, path, h in zip(_PAIR_PLANS, self.paths, self.hiddens):
            print(f"  {name:12s}: {path} ({h}h)")
        print()
