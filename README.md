# shōbench

A benchmark of agents from shared initial conditions: what they do, and when they
choose to stop.

Every agent gets the same starting state and the same stream of tasks, served by
[shōgym](https://github.com/shojin-lab/shogym); the record is what the server observed,
never what the agent reports. The quantities of interest are behavioral: what an agent
does with identical affordances, and when it decides it is finished.

Early evidence for the shape of this benchmark came from shōgym's own quickstart
verifications: on the same three-task queue with the same prompt, one model completed
every task, another quit after one and summarized confidently, and a third drained two
tasks it never played. Same initial conditions; the difference was the agent.

This repo will hold the benchmark definitions, the deployment and evaluation code, and
the analysis. The replication data for published results lives in
[shōrep](https://github.com/shojin-lab/shorep).
