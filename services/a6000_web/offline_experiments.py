from __future__ import annotations

import base64
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable


PROMPT_TYPES = {"clear", "ambiguous", "partial"}
EXPERIMENT_TYPES = {
    "main",
    "non_interactive_baselines",
    "interactive_baselines",
    "efe_ablation",
    "pilot",
}
TRIAL_METHODS = {
    "proposed_efe",
    "top_score",
    "random_candidate",
    "vlm_direct",
    "first_question",
    "random_question",
    "vlm_best_question",
}
TRIAL_OUTCOMES = {"correct", "wrong", "unresolved", "target_pruned", "aborted"}


def now_epoch_s() -> float:
    return time.time()


def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def slug(value: str, fallback: str = "item") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", (value or "").strip()).strip("._")
    return cleaned or fallback


def data_url_to_bytes(data_url: str) -> tuple[bytes, str]:
    header, sep, payload = (data_url or "").partition(",")
    if not sep:
        raise ValueError("invalid data URL")
    mime = "application/octet-stream"
    if header.startswith("data:") and ";base64" in header:
        mime = header[len("data:") : header.index(";base64")]
    return base64.b64decode(payload), mime


def bytes_to_data_url(data: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def compact_session_view(view: Dict[str, Any]) -> Dict[str, Any]:
    hidden = {"observation_image_data_url", "candidate_overlay_data_url", "final_image_data_url"}
    return {key: value for key, value in view.items() if key not in hidden}


class OfflineExperimentStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    def _exp_dir(self, experiment_id: str) -> Path:
        return self.root / slug(experiment_id, "experiment")

    def _scenes_dir(self, experiment_id: str) -> Path:
        return self._exp_dir(experiment_id) / "scenes"

    def _trials_dir(self, experiment_id: str) -> Path:
        return self._exp_dir(experiment_id) / "trials"

    def _experiment_path(self, experiment_id: str) -> Path:
        return self._exp_dir(experiment_id) / "experiment.json"

    def _scene_dir(self, experiment_id: str, scene_id: str) -> Path:
        return self._scenes_dir(experiment_id) / slug(scene_id, "scene")

    def _scene_path(self, experiment_id: str, scene_id: str) -> Path:
        return self._scene_dir(experiment_id, scene_id) / "scene.json"

    def _trial_path(self, experiment_id: str, trial_id: str) -> Path:
        return self._trials_dir(experiment_id) / f"{slug(trial_id, 'trial')}.json"

    def create_experiment(
        self,
        *,
        experiment_id: str | None,
        name: str,
        experiment_type: str,
        notes: str = "",
    ) -> Dict[str, Any]:
        exp_id = slug(experiment_id or f"offline_{stamp()}_{uuid.uuid4().hex[:6]}", "experiment")
        if experiment_type not in EXPERIMENT_TYPES:
            raise ValueError(f"unknown experiment_type: {experiment_type}")
        path = self._experiment_path(exp_id)
        if path.exists():
            record = self.read_experiment(exp_id)
            record.update({"name": name or record.get("name") or exp_id, "experiment_type": experiment_type, "notes": notes})
            record["updated_at_epoch_s"] = now_epoch_s()
        else:
            record = {
                "experiment_id": exp_id,
                "name": name or exp_id,
                "experiment_type": experiment_type,
                "notes": notes,
                "created_at_epoch_s": now_epoch_s(),
                "updated_at_epoch_s": now_epoch_s(),
            }
        self._exp_dir(exp_id).mkdir(parents=True, exist_ok=True)
        self._scenes_dir(exp_id).mkdir(parents=True, exist_ok=True)
        self._trials_dir(exp_id).mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        self.append_event(exp_id, {"event": "experiment_upserted", "experiment": record})
        return record

    def read_experiment(self, experiment_id: str) -> Dict[str, Any]:
        path = self._experiment_path(experiment_id)
        if not path.exists():
            raise FileNotFoundError(f"experiment not found: {experiment_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def list_experiments(self) -> list[Dict[str, Any]]:
        out = []
        for path in sorted(self.root.glob("*/experiment.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                record["metrics"] = self.metrics(record["experiment_id"])
                out.append(record)
            except Exception:
                continue
        return out

    def save_scene(
        self,
        *,
        experiment_id: str,
        scene_id: str,
        scene_type: str,
        object_categories: list[str],
        notes: str,
        image_bytes: bytes,
        mime_type: str,
        observation_id: str | None,
        observation_source: str,
        observation_metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        self.read_experiment(experiment_id)
        scene_dir = self._scene_dir(experiment_id, scene_id)
        scene_dir.mkdir(parents=True, exist_ok=True)
        image_ext = "jpg" if mime_type in {"image/jpeg", "image/jpg"} else "png"
        image_path = scene_dir / f"observation.{image_ext}"
        image_path.write_bytes(image_bytes)
        record = {
            "experiment_id": experiment_id,
            "scene_id": scene_id,
            "scene_type": scene_type,
            "object_categories": object_categories,
            "notes": notes,
            "observation_id": observation_id,
            "observation_source": observation_source,
            "observation_mime_type": mime_type,
            "observation_path": str(image_path),
            "observation_metadata": observation_metadata or {},
            "updated_at_epoch_s": now_epoch_s(),
        }
        if not self._scene_path(experiment_id, scene_id).exists():
            record["created_at_epoch_s"] = record["updated_at_epoch_s"]
        self._scene_path(experiment_id, scene_id).write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        self.append_event(experiment_id, {"event": "scene_saved", "scene": record})
        return self.scene_view(experiment_id, scene_id)

    def scene_view(self, experiment_id: str, scene_id: str, *, include_image: bool = True) -> Dict[str, Any]:
        path = self._scene_path(experiment_id, scene_id)
        if not path.exists():
            raise FileNotFoundError(f"scene not found: {scene_id}")
        record = json.loads(path.read_text(encoding="utf-8"))
        if include_image:
            image_path = Path(record["observation_path"])
            record["observation_image_data_url"] = bytes_to_data_url(
                image_path.read_bytes(),
                record.get("observation_mime_type") or "image/png",
            )
        return record

    def list_scenes(self, experiment_id: str) -> list[Dict[str, Any]]:
        base = self._scenes_dir(experiment_id)
        if not base.exists():
            return []
        scenes = []
        for path in sorted(base.glob("*/scene.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            try:
                scenes.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return scenes

    def write_trial(self, experiment_id: str, trial: Dict[str, Any]) -> Dict[str, Any]:
        self.read_experiment(experiment_id)
        trial["experiment_id"] = experiment_id
        trial["updated_at_epoch_s"] = now_epoch_s()
        self._trials_dir(experiment_id).mkdir(parents=True, exist_ok=True)
        self._trial_path(experiment_id, trial["trial_id"]).write_text(
            json.dumps(trial, indent=2, default=str),
            encoding="utf-8",
        )
        index_path = self._exp_dir(experiment_id) / "trials.jsonl"
        with index_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "updated_at_epoch_s": trial["updated_at_epoch_s"],
                        "trial_id": trial["trial_id"],
                        "scene_id": trial.get("scene_id"),
                        "prompt_type": trial.get("prompt_type"),
                        "method": trial.get("method"),
                        "status": trial.get("status"),
                        "outcome": trial.get("outcome"),
                        "record_path": str(self._trial_path(experiment_id, trial["trial_id"])),
                    },
                    default=str,
                )
                + "\n"
            )
        return trial

    def read_trial(self, experiment_id: str, trial_id: str) -> Dict[str, Any]:
        path = self._trial_path(experiment_id, trial_id)
        if not path.exists():
            raise FileNotFoundError(f"trial not found: {trial_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def list_trials(self, experiment_id: str) -> list[Dict[str, Any]]:
        base = self._trials_dir(experiment_id)
        if not base.exists():
            return []
        trials = []
        for path in sorted(base.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            try:
                trials.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return trials

    def append_event(self, experiment_id: str, event: Dict[str, Any]) -> None:
        exp_dir = self._exp_dir(experiment_id)
        exp_dir.mkdir(parents=True, exist_ok=True)
        event = dict(event)
        event.setdefault("experiment_id", experiment_id)
        event.setdefault("event_id", str(uuid.uuid4()))
        event.setdefault("created_at_epoch_s", now_epoch_s())
        with (exp_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, default=str) + "\n")

    @staticmethod
    def _mean(values: Iterable[float]) -> float | None:
        vals = [float(value) for value in values]
        if not vals:
            return None
        return sum(vals) / len(vals)

    def metrics(self, experiment_id: str) -> Dict[str, Any]:
        try:
            trials = self.list_trials(experiment_id)
        except Exception:
            trials = []

        def summarize(items: list[Dict[str, Any]]) -> Dict[str, Any]:
            total = len(items)
            finished = [item for item in items if item.get("outcome") in TRIAL_OUTCOMES]
            correct = sum(1 for item in finished if item.get("outcome") == "correct")
            wrong = sum(1 for item in finished if item.get("outcome") == "wrong")
            unresolved = sum(1 for item in finished if item.get("outcome") == "unresolved")
            pruned = sum(1 for item in finished if item.get("outcome") == "target_pruned")
            evaluated = max(len(finished), 1)
            return {
                "total": total,
                "finished": len(finished),
                "active": sum(1 for item in items if item.get("status") in {"active", "awaiting_answer", "resolved"}),
                "correct": correct,
                "wrong": wrong,
                "unresolved": unresolved,
                "target_pruned": pruned,
                "resolution_success_rate": correct / evaluated,
                "wrong_object_rate": wrong / evaluated,
                "unresolved_rate": unresolved / evaluated,
                "target_pruned_rate": pruned / evaluated,
                "mean_questions": self._mean(item.get("question_count", 0) for item in finished),
                "avg_latency_s": self._mean(item.get("latency_s", 0.0) for item in finished if item.get("latency_s") is not None),
            }

        by_method = {}
        for method in sorted({str(item.get("method") or "") for item in trials if item.get("method")}):
            by_method[method] = summarize([item for item in trials if item.get("method") == method])

        by_prompt_type = {}
        for prompt_type in sorted({str(item.get("prompt_type") or "") for item in trials if item.get("prompt_type")}):
            by_prompt_type[prompt_type] = summarize([item for item in trials if item.get("prompt_type") == prompt_type])

        return {
            "experiment_id": experiment_id,
            "overall": summarize(trials),
            "by_method": by_method,
            "by_prompt_type": by_prompt_type,
        }
