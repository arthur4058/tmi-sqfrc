"""Run the simplified SORF model selected by component pruning.

The final model keeps the real-observation relational expert and the
validation-constrained probability fusion.  Observation dropping and the
sparse-view consistency losses are disabled because the matched ablation
shows that they do not improve the five-rate or low-rate aggregate results.

This wrapper intentionally leaves ``run_v74_five_rate_multiseed.py`` intact
so that the historical full-model experiments remain reproducible.
"""

from __future__ import annotations

import sys

from scripts.run_v74_five_rate_multiseed import main


def set_default(flag: str, value: str) -> None:
    """Append a default only when the caller did not provide the option."""
    if flag not in sys.argv:
        sys.argv.extend((flag, value))


if __name__ == "__main__":
    set_default("--drop-probability", "0")
    set_default("--drop-weight", "0")
    set_default("--consistency-weight", "0")
    set_default("--output-tag", "sorf_final")
    main()
