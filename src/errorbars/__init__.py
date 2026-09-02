"""errorbars -- the statistics your eval harness doesn't do.

Your eval says prompt B beat prompt A by 4 points. This tells you whether that is real,
or how big a run it would take for it to be.

Three commands:

* ``errorbars power`` -- what a run of a given size can and cannot detect. No config, no
  network, no API key.
* ``errorbars analyze <results-dir>`` -- paired effects, bootstrap intervals, Holm
  correction and a significant/null/underpowered verdict over any CXS results directory,
  including one another tool wrote.
* ``errorbars run <experiment.yaml>`` -- a small intervention-grid runner with a
  mandatory control arm.

This is a feature-sized library, not a platform. It does not run evals and does not
replace promptfoo or Inspect AI.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
