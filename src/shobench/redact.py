"""Keeping provisioned credential values out of the artifacts a cell publishes.

Every v0 harness gets a shell and full internet, and a rollout whose whole instruction is to
improve itself has an obvious reason to read its own environment and its own auth file. What it
reads is echoed verbatim into the leg's trace, and the runner then copies tails of that output
into ``legs.json``, the manifest's stop evidence, and the results JSON. Nothing between the
agent's ``env`` call and the published file used to look at the bytes, so a single curious
command was enough to put a live token into an artifact meant to be shared.

The runner is the party that can fix this, because it is the party that provisioned the
credential: it knows the exact strings it put into the container's environment and the exact
strings in the auth file it copied into the cell's HOME. So the rule here is exact-value
replacement, not pattern matching. A regex for "things that look like tokens" is both leakier
(it misses the shapes it was not written for) and more destructive (it eats the agent's own
prose); an exact value either appears or it does not.

Three properties this deliberately has:

- **It works on bytes.** A trace is written by another process and can hold anything; decoding
  it to redact it would risk changing bytes that carry no secret at all. Replacing byte
  sequences is lossless everywhere else in the file.
- **It covers cheap encodings.** An agent that base64s or url-encodes its environment before
  printing it is not doing anything exotic, so those forms are replaced too. Anything more
  elaborate (a cipher, a split across lines) is out of reach of any redactor and is left to the
  egress record and the reviewer.
- **It never logs what it protects.** Nothing in this module prints, raises, or returns a
  secret; failures are reported by count.

A separate change is adding reasoning-signature scrubbing at this same boundary. The two are
independent by construction and land in different call sites; when both are in, folding them
into one pass over each artifact is the obvious follow-up.
"""

from __future__ import annotations

import base64
import json
import urllib.parse
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

# What a redacted value is replaced by. Fixed, so a reader of an artifact can tell redaction
# from an agent that happened to write something odd, and so two artifacts stay comparable.
MARKER = "[shobench-redacted]"

# The shortest string treated as a secret. Credential material is long and unbroken; a short
# value is far more likely to be a schema token ("oauth", "chatgpt") whose replacement would
# corrupt the artifact without protecting anything.
MIN_SECRET_LEN = 20


def _encodings(value: str) -> list[str]:
    """The forms of one secret worth replacing: the value, and the cheap ways it gets printed.

    JSON escaping is what a trace does to it, url-encoding is what a query string does, and
    base64 is what a shell one-liner does. All four are a substring search away; nothing beyond
    them is, so nothing beyond them is claimed.
    """
    raw = value.encode("utf-8")
    forms = {
        value,
        json.dumps(value)[1:-1],
        urllib.parse.quote(value, safe=""),
        base64.b64encode(raw).decode("ascii"),
        base64.urlsafe_b64encode(raw).decode("ascii"),
        base64.b64encode(raw).decode("ascii").rstrip("="),
        base64.urlsafe_b64encode(raw).decode("ascii").rstrip("="),
    }
    return [form for form in forms if len(form) >= MIN_SECRET_LEN]


def _is_secret_shaped(value: object) -> bool:
    """Is this the kind of string a credential file holds, rather than a field name or a mode?

    Length plus the absence of whitespace. Both matter: the length keeps ``"oauth"`` and
    ``"chatgpt"`` out, and the whitespace rule keeps a sentence out of the needle list, since a
    sentence that appeared in two places would be replaced in both.
    """
    return isinstance(value, str) and len(value) >= MIN_SECRET_LEN and not any(
        char.isspace() for char in value
    )


def secret_strings(body: Any) -> set[str]:
    """Every secret-shaped string leaf of a parsed credential file.

    Schema-blind on purpose. The three harnesses nest their tokens differently and change the
    nesting between versions, so a walk that names ``tokens.access_token`` protects exactly the
    version it was written against. Walking every leaf and filtering by shape protects the ones
    nobody has seen yet, and over-collecting an account id costs a redacted account id.
    """
    found: set[str] = set()
    if isinstance(body, dict):
        for value in body.values():
            found |= secret_strings(value)
    elif isinstance(body, list):
        for value in body:
            found |= secret_strings(value)
    elif _is_secret_shaped(body):
        found.add(str(body))
    return found


def secrets_in_file(path: Path) -> set[str]:
    """The secret-shaped strings in a credential file the runner placed, or nothing.

    A file that will not parse is not an error here. The runner copies whatever the host's login
    minted, and a shape this cannot read is exactly the shape whose secrets it cannot name; the
    caller still redacts everything it does know. Failing instead would take down a cell over a
    credential file the harness itself is perfectly happy with.
    """
    try:
        return secret_strings(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return set()


class Redactor:
    """Exact-value replacement over the artifacts one cell publishes.

    Built once per cell from what the runner provisioned, and then applied at every point where
    a durable artifact is written. Empty is a valid state and means "this cell provisioned no
    secret this runner can name", which is the truth for a harness whose whole credential is a
    file this runner never parsed; the calls all become no-ops rather than being skipped, so no
    call site has to know.
    """

    def __init__(self, values: Iterable[str] = ()) -> None:
        # The values are kept as well as their forms, so a later one can be folded in without the
        # caller having to remember what this was built from. They are never read back out.
        self._values = frozenset(value for value in values if _is_secret_shaped(value))
        needles: set[str] = set()
        for value in self._values:
            needles.update(_encodings(value))
        # Longest first, so a form that contains a shorter form is replaced whole rather than
        # left holding a marker in the middle of itself.
        self._needles = sorted(needles, key=len, reverse=True)
        self._bytes = [(needle.encode("utf-8"), MARKER.encode("utf-8")) for needle in self._needles]

    def __bool__(self) -> bool:
        return bool(self._needles)

    @property
    def count(self) -> int:
        """How many distinct forms this will replace. Never the forms themselves."""
        return len(self._needles)

    def extended(self, values: Iterable[str]) -> Redactor:
        """A redactor watching everything this one watches, plus whatever ``values`` names.

        A credential is not a constant for the life of a cell. A file-backed OAuth client
        refreshes an expired token and writes the new one back over the file the runner seeded,
        so the value that has to be replaced at the end of a leg is not always the value that
        existed when the cell started. Both have to be covered: the old one is still in whatever
        was written before the refresh, and the new one is what everything after it will carry.
        Extending rather than replacing is what keeps the first of those true.

        A new object rather than a mutation, because a cell's eval phase redacts from several
        threads at once. Swapping a reference is atomic and every needle either object holds is a
        needle already in use, so a leg finishing against the older one is redacted by it in full
        rather than against a list being rebuilt underneath it.
        """
        fresh = {value for value in values if _is_secret_shaped(value)} - self._values
        return self if not fresh else Redactor(self._values | fresh)

    def text(self, body: str) -> str:
        for needle in self._needles:
            body = body.replace(needle, MARKER)
        return body

    def data(self, body: bytes) -> bytes:
        for needle, marker in self._bytes:
            body = body.replace(needle, marker)
        return body

    def json(self, body: Any) -> Any:
        """A copy of a JSON-shaped object with every secret replaced, keys included.

        Keys are walked as well as values because a harness that reports usage keyed by
        something it read from the environment would otherwise publish it as a key.
        """
        if isinstance(body, dict):
            return {self.json(key): self.json(value) for key, value in body.items()}
        if isinstance(body, list):
            return [self.json(item) for item in body]
        if isinstance(body, tuple):
            return tuple(self.json(item) for item in body)
        if isinstance(body, str):
            return self.text(body)
        return body

    def file(self, path: Path) -> bool:
        """Rewrite one artifact in place with its secrets replaced. True when it changed.

        Byte-level and idempotent, so a trace a later leg appends to can be passed through this
        again. Rewritten only when something actually matched, which keeps the common case a
        read rather than a read and a write, and best-effort on the way out: a trace that cannot
        be rewritten must not take a running cell down, and the artifacts the runner itself
        writes are redacted through :meth:`json` regardless of what happened here.
        """
        if not self._needles:
            return False
        try:
            body = path.read_bytes()
        except OSError:
            return False
        cleaned = self.data(body)
        if cleaned == body:
            return False
        try:
            path.write_bytes(cleaned)
        except OSError:
            return False
        return True


def redactor_for(
    *, environment: Mapping[str, str] = {}, credential_files: Iterable[Path] = ()
) -> Redactor:
    """The redactor for one cell: what the runner put in the environment, and what it copied.

    Both halves matter and neither covers the other. The environment half is the token a harness
    reads directly, which the runner holds as a value. The file half is the auth file the runner
    copied from the host into the cell's isolated HOME, whose contents the runner never needs to
    look at otherwise; it parses it here so the values inside it can be named at the boundary,
    and keeps them nowhere else.
    """
    values = set(environment.values())
    for path in credential_files:
        values |= secrets_in_file(path)
    return Redactor(values)


__all__ = [
    "MARKER",
    "MIN_SECRET_LEN",
    "Redactor",
    "redactor_for",
    "secret_strings",
    "secrets_in_file",
]
