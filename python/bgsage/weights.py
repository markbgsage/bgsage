"""Weight configuration for the 5-NN game plan strategy."""

from __future__ import annotations

import os
from dataclasses import dataclass


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

    @classmethod
    def default(cls) -> WeightConfig:
        """Stage 5 models bundled with the engine.

        Looks for weight files in bgsage/models/ relative to this package.
        """
        models_dir = os.path.join(
            os.path.dirname(__file__), os.pardir, os.pardir, "models"
        )
        models_dir = os.path.normpath(models_dir)
        return cls(
            purerace=os.path.join(models_dir, "sl_s5_purerace.weights.best"),
            racing=os.path.join(models_dir, "sl_s5_racing.weights.best"),
            attacking=os.path.join(models_dir, "sl_s5_attacking.weights.best"),
            priming=os.path.join(models_dir, "sl_s5_priming.weights.best"),
            anchoring=os.path.join(models_dir, "sl_s5_anchoring.weights.best"),
        )
