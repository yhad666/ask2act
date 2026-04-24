# ask2act

This repository currently contains two actively used parts of Ask2Act:

- `simulation/`: the current grasp / motion / geometry stack
- `services/a6000_web/`: the A6000-side browser, detection, disambiguation, and orchestration service
- `services/a6000_sim/`: a simulation-only browser and orchestration workspace kept separate from the real-robot path
- `real/stretch_transport/`: the Stretch-side ZMQ server that the A6000 calls for observation and execution

The merged A6000 service now owns:

- browser UI
- instruction phrase extraction
- GroundingDINO candidate generation
- VLM / EFE clarification
- resolved-target to bbox handoff into the local grasp pipeline
- direct Stretch transport calls for observation and execution

See [`services/a6000_web/README.md`](./services/a6000_web/README.md) for the real-robot-side A6000 service.

See [`real/stretch_transport/README.md`](./real/stretch_transport/README.md) for the Stretch-side one-command server.

See [`services/a6000_sim/README.md`](./services/a6000_sim/README.md) for the dedicated simulation thesis-testing workspace.

Current active target:

- one table
- one cup
- one head-camera-first grasp pipeline

See [`simulation/README.md`](./simulation/README.md) for setup, status, and usage.
