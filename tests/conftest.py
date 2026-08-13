"""What the suite has to settle before anything imports a client that would go out.

Two libraries here read a setting once, at import, and act on it for the life of the process.
``huggingface_hub`` reads ``HF_HUB_OFFLINE``, and ``datasets`` takes its own offline flag from that
same constant. FastMCP reads ``FASTMCP_CHECK_FOR_UPDATES`` into its settings object, and with it
left alone the first server this suite starts asks PyPI whether a newer FastMCP exists, which is a
real request out of a suite that claims to make none. A test that sets either variable sets it too
late. They are set here instead: pytest imports this file before it imports a test module, and
before any of them import either library.

That is what makes this suite's verdict a property of the machine it runs on rather than of
whether that machine has a network, and it is what the workflow's claim about the suite rests on.
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
