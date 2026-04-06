# Ask2Act Grasp Package

This package is the new primary implementation path for the `ask2act` simulation grasp stack.

## Modules

- `scene/`: runtime MuJoCo scene generation for the table and cup
- `perception/`: head alignment and point-cloud generation
- `grasp/`: Contact-GraspNet wrapper plus fallback grasp generation and selection
- `planning/`: planner interface with the current geometry-aware fallback backend
- `execution/`: staged grasp execution orchestration
- `config/`: scene and grasp parameters
- `tests/`: lightweight regression tests for the new package

## Design Intent

The interfaces are intentionally shaped so we can later:

- plug in upstream VLM outputs such as `target_bbox_2d` and object masks
- swap the fallback grasp generator for real Contact-GraspNet inference
- swap the fallback planner for MoveIt2
- keep most of the orchestration code unchanged when moving toward the real robot

## Current Reality

The package is publishable and runnable, but not yet feature-complete:

- the code structure is now clean enough for GitHub handoff
- the runtime scene and pipeline entrypoint are stable
- the current VM still struggles with head motion once RGB-D rendering is enabled
- that environment bottleneck is the main remaining blocker before full end-to-end grasp execution
