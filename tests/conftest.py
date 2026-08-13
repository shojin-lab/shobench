"""What the suite has to settle before anything imports a client that would go out.

Both libraries below read their setting once, at import, and act on it for the life of the
process: ``huggingface_hub`` reads ``HF_HUB_OFFLINE`` and ``datasets`` takes its offline flag from
that same constant, while FastMCP reads ``FASTMCP_CHECK_FOR_UPDATES`` into its settings object and
without it asks PyPI about new releases the first time this suite starts a server. A test that
sets either variable sets it too late, which is why they are set here: pytest imports this file
before it imports a test module, and before any of them import either library.

Here and nowhere else. Every way into this suite comes through this file, including a developer
running pytest by hand, so a second copy of these pins in the workflow would only drift.
"""

from __future__ import annotations

import os
import sys

# The library that reads the setting, and the setting that keeps it home.
_PINS = {
    "huggingface_hub": ("HF_HUB_OFFLINE", "1"),
    "fastmcp": ("FASTMCP_CHECK_FOR_UPDATES", "off"),
}

for _module, (_name, _value) in _PINS.items():
    if _module in sys.modules:
        # It has read the setting already, so setting the variable now would only look like a fix.
        raise RuntimeError(
            f"{_module} was imported before conftest could set {_name}, and the suite cannot "
            "promise it will stay off the network"
        )
    os.environ[_name] = _value
