from __future__ import annotations

import json
from pathlib import Path

from pipeline_types import TargetSelection


class Layer0TargetProvider:
    def load(self) -> TargetSelection:
        raise NotImplementedError


class ManualJsonTargetProvider(Layer0TargetProvider):
    def __init__(self, json_path: str | Path):
        self.json_path = Path(json_path).resolve()

    def load(self) -> TargetSelection:
        payload = json.loads(self.json_path.read_text(encoding="utf-8"))
        return TargetSelection.from_dict(payload, base_dir=self.json_path.parent)


class RemoteLayer0Provider(Layer0TargetProvider):
    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def load(self) -> TargetSelection:
        raise NotImplementedError(
            "RemoteLayer0Provider is a placeholder only in v1. "
            "Use ManualJsonTargetProvider until the remote layer-0 service is available."
        )
