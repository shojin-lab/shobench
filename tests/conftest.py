"""What the suite has to settle before anything imports a Hub client.

``huggingface_hub`` reads ``HF_HUB_OFFLINE`` once, at import, and ``datasets`` takes its own
offline flag from that same constant. A test that sets the variable sets it too late to stop
anything, so it is set here instead: pytest imports this file before it imports a test module,
and before any of them import the Hub client. With it set, a test that reaches the Hub reads the
local cache or raises; it cannot download. That is what makes this suite's verdict a property of
the machine it runs on rather than of whether that machine has a network.
"""

from __future__ import annotations

import os
import sys

if "huggingface_hub" in sys.modules:
    # The constant is already read, so setting the variable now would only look like a fix.
    raise RuntimeError(
        "huggingface_hub was imported before conftest could put it offline; the suite cannot "
        "promise it will not download"
    )

os.environ["HF_HUB_OFFLINE"] = "1"
