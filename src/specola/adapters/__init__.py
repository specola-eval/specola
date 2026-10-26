"""
specola.adapters — bundled simulator adapters.

Each module self-registers via specola.registry on import and is
exposed through the ``specola.adapters`` entry-point group. Adapters
only translate simulator state into canonical MonitorObservation
objects; simulator control stays in experiment code.
"""

from __future__ import annotations
