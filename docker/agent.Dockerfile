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

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates ripgrep curl jq procps \
        python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

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
ARG PRIME_AGENT_VERSION=0.7.1
RUN PRIME_AGENT_INSTALLER_PLAIN=1 \
    PRIME_AGENT_BOOTSTRAP_TOOLS_ON_INSTALL=1 \
    ${PRIME_AGENT_VERSION:+PRIME_AGENT_VERSION=$PRIME_AGENT_VERSION} \
    sh -c 'curl -fsSL https://app.primeintellect.ai/prime-agent/install.sh | sh'

WORKDIR /work
CMD ["bash"]
