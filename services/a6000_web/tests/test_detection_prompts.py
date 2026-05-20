from __future__ import annotations

import io

from PIL import Image

from services.a6000_web.detection import GroundingDinoDetector, _label_position_for_bbox, _rect_overlap_area
from services.a6000_web.phrase_extractor import InstructionPhraseExtractor
from services.a6000_web.schemas import Candidate


BROAD_CUP_PROMPT = "cup . bottle . spoon . fork . knife . plate ."
BROAD_MUG_PROMPT = "mug . cup . bottle . spoon . fork . knife . plate ."
FORK_SAME_CLASS_PROMPT = "fork . forks . plastic fork . table fork . dining fork ."
FORK_CONTEXT_PROMPT = "fork . cup . bottle . spoon . knife . plate ."
YELLOW_FORK_CONTEXT_PROMPT = "yellow fork . cup . bottle . spoon . fork . knife . plate ."
LEFT_FORK_CONTEXT_PROMPT = "left fork . cup . bottle . spoon . fork . knife . plate ."
RED_CUP_CONTEXT_PROMPT = "red cup . cup . bottle . spoon . fork . knife . plate ."
ORANGE_CUP_CONTEXT_PROMPT = "orange cup . cup . bottle . spoon . fork . knife . plate ."
LEFT_BLUE_CUP_CONTEXT_PROMPT = "left blue cup . cup . bottle . spoon . fork . knife . plate ."
LEFT_CUP_CONTEXT_PROMPT = "left cup . cup . bottle . spoon . fork . knife . plate ."
LEFTMOST_MUG_CONTEXT_PROMPT = "leftmost mug . cup . bottle . spoon . fork . knife . plate ."


def _image_bytes() -> bytes:
    image = Image.new("RGB", (16, 16), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _candidate(candidate_id: str, label: str) -> Candidate:
    idx = int(candidate_id.rsplit("_", 1)[-1])
    return Candidate(
        candidate_id=candidate_id,
        display_id=idx,
        label=label,
        score=0.5,
        bbox_xyxy=[float(idx), 1.0, float(idx + 1), 2.0],
        mask_rle=None,
    )


class FakeDetector(GroundingDinoDetector):
    def __init__(self, responses: dict[str, list[Candidate]]) -> None:
        super().__init__(phrase_extractor=InstructionPhraseExtractor())
        self.responses = responses
        self.prompts: list[str] = []

    def _ensure_model(self) -> None:
        return None

    def _run_grounding_dino(self, prepared: Image.Image, prompt: str, **kwargs) -> list[Candidate]:
        self.prompts.append(prompt)
        return list(self.responses.get(prompt, []))

    def render_overlay(self, *args, **kwargs) -> str:
        return "data:image/png;base64,"


def test_attribute_relaxed_hit_is_not_overwritten_by_broad_fallback() -> None:
    blue_cup = _candidate("cand_001", "blue cup")
    generic_cups = [_candidate("cand_001", "cup"), _candidate("cand_002", "cup")]
    detector = FakeDetector(
        {
            "left blue cup .": [],
            "blue cup .": [blue_cup],
            BROAD_CUP_PROMPT: generic_cups,
        }
    )

    result = detector.detect(_image_bytes(), "Pick up the left blue cup.")

    assert result.detection_prompt == "blue cup ."
    assert result.candidates == [blue_cup]
    assert detector.prompts == ["left blue cup .", "blue cup ."]


def test_strict_spatial_and_attribute_hits_are_not_overwritten_by_broad_fallback() -> None:
    for instruction, strict_prompt in [
        ("Pick up the most left cup.", "leftmost cup ."),
        ("Pick up the most right cup.", "rightmost cup ."),
        ("Pick up the front cup.", "front cup ."),
    ]:
        strict_hit = _candidate("cand_001", strict_prompt[:-2])
        generic_cups = [_candidate("cand_001", "cup"), _candidate("cand_002", "cup")]
        detector = FakeDetector(
            {
                strict_prompt: [strict_hit],
                BROAD_CUP_PROMPT: generic_cups,
            }
        )

        result = detector.detect(_image_bytes(), instruction)

        assert result.detection_prompt == strict_prompt
        assert result.candidates == [strict_hit]
        assert detector.prompts == [strict_prompt]


def test_attribute_object_recall_uses_base_object_prompt_when_strict_has_no_candidates() -> None:
    generic_cups = [_candidate("cand_001", "cup"), _candidate("cand_002", "cup")]
    detector = FakeDetector(
        {
            "yellow cup .": [],
            "cup .": generic_cups,
        }
    )

    result = detector.detect(_image_bytes(), "Pick up the yellow cup.", enable_attribute_object_recall=True)

    assert result.detection_prompt == "cup ."
    assert result.candidates == generic_cups
    assert detector.prompts == ["yellow cup .", "cup ."]


def test_attribute_object_recall_keeps_single_strict_candidate_by_default() -> None:
    yellow_cup = _candidate("cand_001", "yellow cup")
    generic_cups = [_candidate("cand_001", "cup"), _candidate("cand_002", "cup")]
    detector = FakeDetector(
        {
            "yellow cup .": [yellow_cup],
            "cup .": generic_cups,
        }
    )

    result = detector.detect(_image_bytes(), "Pick up the yellow cup.", enable_attribute_object_recall=True)

    assert result.detection_prompt == "yellow cup ."
    assert result.candidates == [yellow_cup]
    assert detector.prompts == ["yellow cup ."]


def test_attribute_object_recall_can_broaden_single_candidate_when_threshold_allows() -> None:
    generic_cups = [_candidate("cand_001", "cup"), _candidate("cand_002", "cup")]
    detector = FakeDetector(
        {
            "yellow cup .": [_candidate("cand_001", "yellow cup")],
            "cup .": generic_cups,
        }
    )

    result = detector.detect(
        _image_bytes(),
        "Pick up the yellow cup.",
        enable_attribute_object_recall=True,
        attribute_object_recall_candidate_threshold=1,
    )

    assert result.detection_prompt == "cup ."
    assert result.candidates == generic_cups
    assert detector.prompts == ["yellow cup .", "cup ."]


def test_attribute_object_recall_keeps_strict_prompt_when_it_finds_multiple_targets() -> None:
    yellow_cups = [_candidate("cand_001", "yellow cup"), _candidate("cand_002", "yellow cup")]
    generic_cups = [_candidate("cand_001", "cup"), _candidate("cand_002", "cup"), _candidate("cand_003", "cup")]
    detector = FakeDetector(
        {
            "yellow cup .": yellow_cups,
            "cup .": generic_cups,
        }
    )

    result = detector.detect(_image_bytes(), "Pick up the yellow cup.", enable_attribute_object_recall=True)

    assert result.detection_prompt == "yellow cup ."
    assert result.candidates == yellow_cups
    assert detector.prompts == ["yellow cup ."]


def test_attribute_object_recall_also_broadens_spatial_partial_prompts() -> None:
    generic_cups = [_candidate("cand_001", "cup"), _candidate("cand_002", "cup")]
    detector = FakeDetector(
        {
            "left cup .": [_candidate("cand_001", "left cup")],
            "cup .": generic_cups,
        }
    )

    result = detector.detect(
        _image_bytes(),
        "Pick up the left cup.",
        enable_attribute_object_recall=True,
        attribute_object_recall_candidate_threshold=1,
    )

    assert result.detection_prompt == "cup ."
    assert result.candidates == generic_cups
    assert detector.prompts == ["left cup .", "cup ."]


def test_attribute_object_recall_is_disabled_by_default() -> None:
    yellow_cup = _candidate("cand_001", "yellow cup")
    generic_cups = [_candidate("cand_001", "cup"), _candidate("cand_002", "cup")]
    detector = FakeDetector(
        {
            "yellow cup .": [yellow_cup],
            BROAD_CUP_PROMPT: generic_cups,
        }
    )

    result = detector.detect(_image_bytes(), "Pick up the yellow cup.")

    assert result.detection_prompt == "yellow cup ."
    assert result.candidates == [yellow_cup]
    assert detector.prompts == ["yellow cup ."]


def test_spatial_attribute_class_uses_attribute_prompt_before_broad_fallback() -> None:
    for instruction, strict_prompt, attribute_prompt in [
        ("Pick up the left blue cup.", "left blue cup .", "blue cup ."),
        ("Pick up the right orange cup.", "right orange cup .", "orange cup ."),
        ("Pick up the back metal cup.", "back metal cup .", "metal cup ."),
    ]:
        attribute_hit = _candidate("cand_001", attribute_prompt[:-2])
        generic_cups = [_candidate("cand_001", "cup"), _candidate("cand_002", "cup")]
        detector = FakeDetector(
            {
                strict_prompt: [],
                attribute_prompt: [attribute_hit],
                BROAD_CUP_PROMPT: generic_cups,
            }
        )

        result = detector.detect(_image_bytes(), instruction)

        assert result.detection_prompt == attribute_prompt
        assert result.candidates == [attribute_hit]
        assert detector.prompts == [strict_prompt, attribute_prompt]


def test_attribute_object_recall_skips_strict_and_relaxed_attribute_prompts() -> None:
    generic_cups = [_candidate("cand_001", "cup"), _candidate("cand_002", "cup")]
    detector = FakeDetector(
        {
            "small red cup .": [],
            "red cup .": [_candidate("cand_001", "red cup")],
            "cup .": generic_cups,
        }
    )

    result = detector.detect(
        _image_bytes(),
        "Pick up the small red cup.",
        enable_attribute_object_recall=True,
        attribute_object_recall_candidate_threshold=1,
    )

    assert result.detection_prompt == "cup ."
    assert result.candidates == generic_cups
    assert detector.prompts == ["small red cup .", "cup ."]


def test_context_prompt_uses_dot_separated_helper_classes_and_filters_to_target() -> None:
    fork = _candidate("cand_001", "fork")
    spoon = _candidate("cand_002", "spoon")
    detector = FakeDetector(
        {
            "fork .": [],
            FORK_SAME_CLASS_PROMPT: [],
            FORK_CONTEXT_PROMPT: [spoon, fork],
        }
    )

    result = detector.detect(_image_bytes(), "Pick up my fork.")

    assert result.detection_prompt == FORK_CONTEXT_PROMPT
    assert result.candidates == [fork]
    assert detector.prompts == ["fork .", FORK_SAME_CLASS_PROMPT, FORK_CONTEXT_PROMPT]
    assert "fork . cup" in FORK_CONTEXT_PROMPT
    assert FORK_CONTEXT_PROMPT.count("fork") == 1


def test_single_fork_candidate_only_expands_to_same_class_aliases() -> None:
    fork = _candidate("cand_001", "fork")
    detector = FakeDetector(
        {
            "fork .": [fork],
            FORK_SAME_CLASS_PROMPT: [],
            FORK_CONTEXT_PROMPT: [_candidate("cand_002", "spoon"), fork],
        }
    )

    result = detector.detect(_image_bytes(), "Pick up my fork.")

    assert result.detection_prompt == "fork ."
    assert result.candidates == [fork]
    assert detector.prompts == ["fork .", FORK_SAME_CLASS_PROMPT]


def test_fork_same_class_aliases_can_recover_more_forks_without_spoon_prompt() -> None:
    strict_fork = _candidate("cand_001", "fork")
    recovered_forks = [strict_fork, _candidate("cand_002", "plastic fork"), _candidate("cand_003", "table fork")]
    detector = FakeDetector(
        {
            "fork .": [strict_fork],
            FORK_SAME_CLASS_PROMPT: recovered_forks,
        }
    )

    result = detector.detect(_image_bytes(), "Pick up my fork.")

    assert result.detection_prompt == FORK_SAME_CLASS_PROMPT
    assert result.candidates == recovered_forks
    assert detector.prompts == ["fork .", FORK_SAME_CLASS_PROMPT]
    assert "spoon" not in FORK_SAME_CLASS_PROMPT


def test_fork_same_class_aliases_are_skipped_when_multiple_strict_forks_are_found() -> None:
    forks = [_candidate("cand_001", "fork"), _candidate("cand_002", "fork")]
    detector = FakeDetector(
        {
            "fork .": forks,
            FORK_SAME_CLASS_PROMPT: [*forks, _candidate("cand_003", "table fork")],
        }
    )

    result = detector.detect(_image_bytes(), "Pick up my fork.")

    assert result.detection_prompt == "fork ."
    assert result.candidates == forks
    assert detector.prompts == ["fork ."]


def test_fork_same_class_aliases_are_only_for_unmodified_fork() -> None:
    yellow_fork = _candidate("cand_001", "yellow fork")
    detector = FakeDetector(
        {
            "yellow fork .": [],
            YELLOW_FORK_CONTEXT_PROMPT: [yellow_fork],
            FORK_SAME_CLASS_PROMPT: [_candidate("cand_002", "fork")],
        }
    )

    result = detector.detect(_image_bytes(), "Pick up the yellow fork.")

    assert result.detection_prompt == YELLOW_FORK_CONTEXT_PROMPT
    assert result.candidates == [yellow_fork]
    assert detector.prompts == ["yellow fork .", YELLOW_FORK_CONTEXT_PROMPT]


def test_fork_same_class_aliases_are_not_used_for_spatial_fork() -> None:
    left_fork = _candidate("cand_001", "left fork")
    detector = FakeDetector(
        {
            "left fork .": [],
            LEFT_FORK_CONTEXT_PROMPT: [left_fork],
            FORK_SAME_CLASS_PROMPT: [_candidate("cand_002", "fork")],
        }
    )

    result = detector.detect(_image_bytes(), "Pick up the left fork.")

    assert result.detection_prompt == LEFT_FORK_CONTEXT_PROMPT
    assert result.candidates == [left_fork]
    assert detector.prompts == ["left fork .", LEFT_FORK_CONTEXT_PROMPT]


def test_context_prompt_can_coexist_with_modified_target_and_base_object() -> None:
    red_cup = _candidate("cand_001", "red cup")
    generic_cup = _candidate("cand_002", "cup")
    detector = FakeDetector(
        {
            "red cup .": [],
            RED_CUP_CONTEXT_PROMPT: [generic_cup, red_cup],
        }
    )

    result = detector.detect(_image_bytes(), "Pick up the red cup.")

    assert result.detection_prompt == RED_CUP_CONTEXT_PROMPT
    assert result.candidates == [red_cup]
    assert detector.prompts == ["red cup .", RED_CUP_CONTEXT_PROMPT]
    assert "red cup . cup" in RED_CUP_CONTEXT_PROMPT


def test_broad_fallback_restores_old_base_term_recall_when_strict_prompts_fail() -> None:
    generic_cups = [_candidate("cand_001", "cup"), _candidate("cand_002", "cup")]
    for instruction, prompts_before_broad in [
        ("Pick up the left cup.", ["left cup .", LEFT_CUP_CONTEXT_PROMPT]),
        ("Pick up the orange cup.", ["orange cup .", ORANGE_CUP_CONTEXT_PROMPT]),
        ("Pick up the left blue cup.", ["left blue cup .", "blue cup .", LEFT_BLUE_CUP_CONTEXT_PROMPT]),
    ]:
        responses = {prompt: [] for prompt in prompts_before_broad}
        responses[BROAD_CUP_PROMPT] = generic_cups
        detector = FakeDetector(responses)

        result = detector.detect(_image_bytes(), instruction)

        assert result.detection_prompt == BROAD_CUP_PROMPT
        assert result.candidates == generic_cups
        assert detector.prompts == [*prompts_before_broad, BROAD_CUP_PROMPT]


def test_broad_fallback_keeps_non_default_object_nouns_from_old_strategy() -> None:
    mug = _candidate("cand_001", "mug")
    detector = FakeDetector(
        {
            "leftmost mug .": [],
            LEFTMOST_MUG_CONTEXT_PROMPT: [],
            BROAD_MUG_PROMPT: [mug],
        }
    )

    result = detector.detect(_image_bytes(), "Pick up the leftmost mug.")

    assert result.detection_prompt == BROAD_MUG_PROMPT
    assert result.candidates == [mug]
    assert detector.prompts == ["leftmost mug .", LEFTMOST_MUG_CONTEXT_PROMPT, BROAD_MUG_PROMPT]


def test_overlay_label_prefers_space_outside_candidate_box() -> None:
    candidate_box = [70.0, 70.0, 110.0, 105.0]
    x, y = _label_position_for_bbox((200, 200), candidate_box, (70, 24), 4, [candidate_box])

    label_box = [float(x), float(y), float(x + 70), float(y + 24)]
    assert _rect_overlap_area(label_box, candidate_box) == 0
    assert y + 24 <= candidate_box[1]


def test_overlay_label_uses_below_when_top_space_is_too_small() -> None:
    candidate_box = [70.0, 5.0, 110.0, 35.0]
    x, y = _label_position_for_bbox((200, 200), candidate_box, (70, 24), 4, [candidate_box])

    label_box = [float(x), float(y), float(x + 70), float(y + 24)]
    assert _rect_overlap_area(label_box, candidate_box) == 0
    assert y >= candidate_box[3] + 4
