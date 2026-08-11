# The agent container: one image, identical across every harness, because the scope makes the
# container image part of the initial conditions. All three CLIs install system-wide under
# /usr/local so the cell's isolated HOME can be mounted over /root without hiding them.
#
# NO CREDENTIAL IS BAKED IN. Every key and token arrives as `docker run -e` at runtime, and the
# container mounts nothing of the serving side, so the provenance directory holding scores and
# held-out answers is unreachable from here even with full Bash and full internet. That is what
# makes "leakage is observed, not gated" scientifically clean: the agent is free, the record is
# not.
FROM node:22-bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
# An inherited NODE_OPTIONS reaches each harness's own Node runtime and has broken launches
# before, so the image clears it and the runner sets it empty again per leg.
ENV NODE_OPTIONS=""

# ripgrep and fd-find are here rather than left to a harness to fetch at first use. prime-agent
# resolves both by looking in its own tools directory and then falling back to PATH, and its own
# tools directory lives under the HOME the runner mounts over, so a system-wide copy is the one
# it can still see. Debian names the fd binary `fdfind`, which is one of the names prime-agent
# already looks for; the symlink is for the agent's own shell, which will type `fd`.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates ripgrep fd-find curl jq procps \
        python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/fdfind /usr/local/bin/fd

# --- the three harnesses, pinned ---------------------------------------------------------
# Versions are pinned because "which harness answered" is part of the record. Bumping one is a
# reviewable commit, never a silent difference between two cells.
ARG CLAUDE_CODE_VERSION=2.1.226
ARG CODEX_VERSION=0.147.0

RUN npm install -g --no-fund --no-audit \
        "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" \
        "@openai/codex@${CODEX_VERSION}" \
    && npm cache clean --force

# prime-agent installs from its vendor script, never from npm: the npm name in its source tree
# installs Pi instead. The script downloads a checksum-verified release tarball and hands it to
# `npm install -g`, so the command still lands in /usr/local. PRIME_AGENT_INSTALLER_PLAIN keeps
# the output greppable in a build log. The version is set rather than left empty: the installer
# reads PRIME_AGENT_VERSION when it has one and resolves the stable channel when it does not, so
# an empty value pins nothing and two images built a week apart could carry different agents.
#
# Both bootstrap variables are set for the same reason. The installer's postinstall preloads the
# search tools only when TOOLS is 1 and builds the IPython kernel venv only when KERNEL is 1;
# left unset, KERNEL is a question the installer asks, and a build with no terminal answers it
# yes on the build's behalf. That default is right and undeclared, which is the same shape of
# problem as an unpinned version: the image would stop baking the kernel the day the prompt's
# fallback changes, and the first measured task would pay for a network package install.
#
# Everything the kernel bootstrap produces is placed OUTSIDE /root, because the runner mounts
# the cell's isolated HOME over all of /root and a bind mount hides whatever the image put
# underneath it. Left at their defaults, prime-agent's kernel venv lands in
# /root/.prime/agent/kernel-venv, uv lands in /root/.local/bin (which is not on PATH), and the
# uv-managed interpreter the venv symlinks to lands in /root/.local/share. All three vanish the
# moment a real cell starts, and prime-agent then reports its one-time setup and fails on the
# missing uv. Verified against this image: unmounted it prints the baked kernel python; with the
# runtime /root bind it prints "uv is required to set up the Python kernel". So a prime_agent
# cell either had a broken IPython tool or paid for a network bootstrap during measured time.
#
# The three variables below are set as ENV rather than only for this RUN, so the build and the
# runtime resolve the same paths. What deliberately stays under the mounted HOME is everything
# mutable and per-cell: settings.json, auth.json, skills/, sessions/. The venv is the opposite,
# an immutable part of the initial conditions, and living in the image layer means every leg
# starts from the same one and any write to it is discarded with the container.
ENV UV_INSTALL_DIR=/usr/local/bin
ENV UV_PYTHON_INSTALL_DIR=/opt/prime-agent/uv-python
ENV PRIME_AGENT_KERNEL_VENV=/opt/prime-agent/kernel-venv

# The version is assigned outright rather than through `${VAR:+VAR=$VAR}`. A shell recognises an
# assignment when it parses the word, and a word that begins with `$` is a command word by then,
# so the conditional form ran the expansion as a program and the build failed with
# "PRIME_AGENT_VERSION=0.7.1: not found". An empty build arg still resolves the stable channel,
# which the installer treats the same as unset, so nothing is lost by dropping the condition.
ARG PRIME_AGENT_VERSION=0.7.1

# The install and its verification are one layer, because a build that installed the agent but
# not its kernel must not be cacheable as a success. The postinstall swallows its own errors, so
# a bootstrap that did not happen otherwise leaves a green build and a cell that pays for the
# bootstrap during measured time. Each check names a way this has actually gone wrong: uv off
# PATH, the venv back under /root where the mount hides it, and a venv with no ipykernel in it.
RUN PRIME_AGENT_INSTALLER_PLAIN=1 \
    PRIME_AGENT_BOOTSTRAP_TOOLS_ON_INSTALL=1 \
    PRIME_AGENT_BOOTSTRAP_KERNEL_ON_INSTALL=1 \
    PRIME_AGENT_VERSION="${PRIME_AGENT_VERSION}" \
    sh -c 'curl -fsSL https://app.primeintellect.ai/prime-agent/install.sh | sh' \
    && test -x /usr/local/bin/uv \
    && test -x "${PRIME_AGENT_KERNEL_VENV}/bin/python" \
    && test ! -e /root/.prime/agent/kernel-venv \
    && test ! -e /root/.local/bin/uv \
    && "${PRIME_AGENT_KERNEL_VENV}/bin/python" -c 'import ipykernel' \
    && command -v fdfind >/dev/null \
    && { [ -z "${PRIME_AGENT_VERSION}" ] \
         || prime-agent --version 2>&1 | grep -qF "${PRIME_AGENT_VERSION}"; }

WORKDIR /work
CMD ["bash"]
