# The passive egress observer. It attaches to a cell's network namespace, prints one TSV line
# per outbound DNS question and TLS client hello, and blocks nothing: tshark reads packets, it
# does not route them. No credential, no mount the agent can see.
FROM debian:stable-slim
ENV DEBIAN_FRONTEND=noninteractive
# tshark's postinst asks whether non-root users may capture; preseed it so the build is
# non-interactive. The container runs as root regardless.
RUN echo "wireshark-common wireshark-common/install-setuid boolean false" | debconf-set-selections \
    && apt-get update \
    && apt-get install -y --no-install-recommends tshark ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENTRYPOINT []
CMD ["tshark", "--version"]
