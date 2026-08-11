# The network-namespace holder. It owns the cell's network identity so the observer can attach
# BEFORE any agent container does, which is what removes the race where a harness's first
# requests go unobserved. It runs nothing and holds no credential.
FROM debian:stable-slim
CMD ["sleep", "infinity"]
