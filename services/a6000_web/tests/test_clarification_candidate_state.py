from __future__ import annotations

import json
from types import SimpleNamespace

from services.a6000_web.clarification import ClarificationEngine
from services.a6000_web.schemas import Candidate, QuestionTurn, SessionState


def _candidate(idx: int, label: str = "cup") -> Candidate:
    return Candidate(
        candidate_id=f"cand_{idx:03d}",
        display_id=idx,
        label=label,
        score=0.5,
        bbox_xyxy=[float(idx), 0.0, float(idx + 1), 1.0],
        mask_rle=None,
    )


def _session(candidates: list[Candidate]) -> SessionState:
    return SessionState(
        session_id="session_test",
        instruction="Pick up the green cup.",
        instruction_phrases=["green cup"],
        detection_prompt="green cup .",
        observation_id="obs_test",
        observation_source="test",
        observation_image_bytes=b"image",
        observation_image_data_url="data:image/png;base64,",
        candidate_overlay_data_url="data:image/png;base64,",
        candidates=candidates,
        vlm_messages=[],
    )


def _response(content: str, finish_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ]
    )


def _legal_protocol(session: SessionState) -> dict:
    candidate_ids = [candidate.candidate_id for candidate in session.candidates]
    return {
        "Head": "probose",
        "Task_ID": 1,
        "Round": 1,
        "Candidate_State": {"plausible": list(candidate_ids), "eliminated": [], "last_removed": []},
        "Question": [
            {
                "id": 1,
                "text": "Is the target object green?",
                "count": {"total": 4, "y": 1, "n": 3},
                "yes_candidates": [candidate_ids[0]],
                "no_candidates": candidate_ids[1:],
            },
            {
                "id": 2,
                "text": "Is the target object orange?",
                "count": {"total": 4, "y": 1, "n": 3},
                "yes_candidates": [candidate_ids[1]],
                "no_candidates": [candidate_ids[0], *candidate_ids[2:]],
            },
            {
                "id": 3,
                "text": "Is it on the left side?",
                "count": {"total": 4, "y": 2, "n": 2},
                "yes_candidates": candidate_ids[:2],
                "no_candidates": candidate_ids[2:],
            },
            {
                "id": 4,
                "text": "Is the target object in the upper half of the visible objects?",
                "count": {"total": 4, "y": 2, "n": 2},
                "yes_candidates": [candidate_ids[0], candidate_ids[2]],
                "no_candidates": [candidate_ids[1], candidate_ids[3]],
            },
        ],
    }


class FakeChatEngine(ClarificationEngine):
    def __init__(self, responses: list[SimpleNamespace], *, gen_max_tokens: int = 256) -> None:
        super().__init__(
            base_url="http://127.0.0.1:9/v1",
            model="dummy",
            system_prompt_path="services/a6000_web/prompts/system_prompt.txt",
            gen_max_tokens=gen_max_tokens,
        )
        self.responses = list(responses)
        self.calls: list[dict] = []

    def _chat(self, messages, max_tokens=None):
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        if not self.responses:
            raise RuntimeError("no fake responses left")
        return self.responses.pop(0)


def test_candidate_state_eliminates_groups_and_preserves_removed_ids() -> None:
    session = _session(
        [
            _candidate(1, "green cup"),
            _candidate(2, "green cup"),
            _candidate(3, "orange cup"),
            _candidate(4, "blue cup"),
        ]
    )
    ClarificationEngine.ensure_candidate_state(session)

    protocol = {
        "Head": "decision",
        "Task_ID": 1,
        "Round": 2,
        "Grasp": "no",
        "Candidate_State": {
            "plausible": ["cand_003", "cand_004"],
            "eliminated": ["cand_001", "cand_002"],
        },
        "Question": [
            {"id": 1, "text": "Is the target cup orange?", "count": {"total": 2, "y": 1, "n": 1}},
            {"id": 2, "text": "Is the target on the left?", "count": {"total": 2, "y": 1, "n": 1}},
            {"id": 3, "text": "Is the target large?", "count": {"total": 2, "y": 1, "n": 1}},
            {"id": 4, "text": "Is the target plastic?", "count": {"total": 2, "y": 1, "n": 1}},
        ],
    }

    ClarificationEngine.sync_candidate_state_from_protocol(session, protocol, source="test")

    assert session.plausible_candidate_ids == ["cand_003", "cand_004"]
    assert session.eliminated_candidate_ids == ["cand_001", "cand_002"]
    assert session.last_removed_candidate_ids == ["cand_001", "cand_002"]


def test_candidate_state_never_readds_eliminated_candidates() -> None:
    session = _session([_candidate(1), _candidate(2), _candidate(3), _candidate(4)])
    session.plausible_candidate_ids = ["cand_003", "cand_004"]
    session.eliminated_candidate_ids = ["cand_001", "cand_002"]

    protocol = {
        "Head": "decision",
        "Task_ID": 1,
        "Round": 3,
        "Grasp": "no",
        "Candidate_State": {
            "plausible": ["cand_001", "cand_003"],
            "eliminated": ["cand_002"],
        },
        "Question": [
            {"id": 1, "text": "Is the target cup orange?", "count": {"total": 1, "y": 1, "n": 0}},
            {"id": 2, "text": "Is the target on the left?", "count": {"total": 1, "y": 1, "n": 0}},
            {"id": 3, "text": "Is the target large?", "count": {"total": 1, "y": 1, "n": 0}},
            {"id": 4, "text": "Is the target plastic?", "count": {"total": 1, "y": 1, "n": 0}},
        ],
    }

    ClarificationEngine.sync_candidate_state_from_protocol(session, protocol, source="test")

    assert session.plausible_candidate_ids == ["cand_003"]
    assert "cand_001" in session.eliminated_candidate_ids
    assert "cand_002" in session.eliminated_candidate_ids
    assert session.last_removed_candidate_ids == ["cand_004"]


def test_resolve_rejects_target_that_is_not_plausible_after_answers() -> None:
    session = _session([_candidate(1), _candidate(2)])
    session.plausible_candidate_ids = ["cand_002"]
    session.eliminated_candidate_ids = ["cand_001"]

    try:
        ClarificationEngine._resolve_target(
            {"Head": "decision", "Grasp": "yes", "Target": {"name": "cand_001"}},
            session,
        )
    except ValueError as exc:
        assert "not plausible" in str(exc)
    else:
        raise AssertionError("resolved an eliminated candidate")


def test_proposed_efe_prefers_diverse_question_but_vlm_best_keeps_vlm_order() -> None:
    session = _session([_candidate(1), _candidate(2), _candidate(3), _candidate(4)])
    session.question_history = [
        QuestionTurn(
            round_index=1,
            question_id=1,
            text="Is the target cup red?",
            answer="n",
            score_py=0.5,
            score_pn=0.5,
            efe_score=-0.693,
        )
    ]
    protocol = {
        "Head": "probose",
        "Task_ID": 1,
        "Round": 1,
        "Candidate_State": {
            "plausible": ["cand_001", "cand_002", "cand_003", "cand_004"],
            "eliminated": [],
        },
        "Question": [
            {
                "id": 1,
                "text": "Is the target cup green?",
                "count": {"total": 4, "y": 2, "n": 2},
                "yes_candidates": ["cand_001", "cand_002"],
                "no_candidates": ["cand_003", "cand_004"],
            },
            {
                "id": 2,
                "text": "Is the target cup on the left?",
                "count": {"total": 4, "y": 2, "n": 2},
                "yes_candidates": ["cand_001", "cand_003"],
                "no_candidates": ["cand_002", "cand_004"],
            },
            {
                "id": 3,
                "text": "Is the target cup blue?",
                "count": {"total": 4, "y": 2, "n": 2},
                "yes_candidates": ["cand_001", "cand_004"],
                "no_candidates": ["cand_002", "cand_003"],
            },
            {
                "id": 4,
                "text": "Is the target cup large?",
                "count": {"total": 4, "y": 2, "n": 2},
                "yes_candidates": ["cand_002", "cand_003"],
                "no_candidates": ["cand_001", "cand_004"],
            },
        ],
    }

    proposed = ClarificationEngine.rank_questions_for_mode(protocol, session, mode="proposed_efe")
    vlm_best = ClarificationEngine.rank_questions_for_mode(protocol, session, mode="vlm_best_question")

    assert proposed[0].id == 2
    assert [question.id for question in vlm_best] == [1, 2, 3, 4]


def test_invalid_vlm_questions_are_replaced_with_legal_backend_questions() -> None:
    session = _session([_candidate(1), _candidate(2), _candidate(3), _candidate(4)])
    ClarificationEngine.ensure_candidate_state(session)
    protocol = {
        "Head": "probose",
        "Task_ID": 1,
        "Round": 1,
        "Candidate_State": {
            "plausible": ["cand_001", "cand_002", "cand_003", "cand_004"],
            "eliminated": [],
        },
        "Question": [
            {"id": 1, "text": "Is it candidate 3?", "count": {"total": 4, "y": 1, "n": 3}},
            {"id": 2, "text": "Which one is it?", "count": {"total": 4, "y": 2, "n": 2}},
        ],
    }

    questions = ClarificationEngine.rank_questions_for_mode(protocol, session, mode="vlm_best_question")

    assert len(questions) == 4
    assert all("candidate" not in question.text.lower() for question in questions)
    assert all(question.text.endswith("?") for question in questions)
    assert all(question.yes_candidate_ids for question in questions)
    assert all(question.no_candidate_ids for question in questions)
    assert all(question.count["y"] + question.count["n"] == question.count["total"] for question in questions)


def test_vlm_question_lists_must_match_current_candidate_state_and_counts() -> None:
    session = _session([_candidate(1), _candidate(2), _candidate(3), _candidate(4)])
    session.plausible_candidate_ids = ["cand_001", "cand_002"]
    session.eliminated_candidate_ids = ["cand_003", "cand_004"]
    protocol = {
        "Head": "decision",
        "Task_ID": 1,
        "Round": 2,
        "Grasp": "no",
        "Candidate_State": {
            "plausible": ["cand_001", "cand_002"],
            "eliminated": ["cand_003", "cand_004"],
        },
        "Question": [
            {
                "id": 1,
                "text": "Is the target object green?",
                "count": {"total": 3, "y": 2, "n": 1},
                "yes_candidates": ["cand_001", "cand_003"],
                "no_candidates": ["cand_002"],
            },
            {
                "id": 2,
                "text": "Is the target object blue?",
                "count": {"total": 2, "y": 1, "n": 1},
                "yes_candidates": ["cand_001"],
                "no_candidates": ["cand_001"],
            },
        ],
    }

    questions = ClarificationEngine.rank_questions_for_mode(protocol, session, mode="vlm_best_question")

    assert len(questions) == 4
    assert all(question.text not in {"Is the target object green?", "Is the target object blue?"} for question in questions)
    assert all(set(question.yes_candidate_ids + question.no_candidate_ids) == {"cand_001", "cand_002"} for question in questions)


def test_questions_that_contradict_answer_history_are_replaced() -> None:
    session = _session(
        [
            _candidate(1, "yellow cup"),
            _candidate(2, "yellow cup"),
            _candidate(3, "blue cup"),
            _candidate(4, "red cup"),
        ]
    )
    session.plausible_candidate_ids = ["cand_001", "cand_002"]
    session.eliminated_candidate_ids = ["cand_003", "cand_004"]
    session.question_history = [
        QuestionTurn(
            round_index=1,
            question_id=1,
            text="Is the target object yellow?",
            answer="y",
            score_py=0.5,
            score_pn=0.5,
            efe_score=-0.693,
        )
    ]
    protocol = {
        "Head": "decision",
        "Task_ID": 1,
        "Round": 2,
        "Grasp": "no",
        "Candidate_State": {
            "plausible": ["cand_001", "cand_002"],
            "eliminated": ["cand_003", "cand_004"],
        },
        "Question": [
            {
                "id": 1,
                "text": "Is the target object blue?",
                "count": {"total": 2, "y": 1, "n": 1},
                "yes_candidates": ["cand_001"],
                "no_candidates": ["cand_002"],
            },
            {
                "id": 2,
                "text": "Is it on the left side?",
                "count": {"total": 2, "y": 1, "n": 1},
                "yes_candidates": ["cand_001"],
                "no_candidates": ["cand_002"],
            },
        ],
    }

    questions = ClarificationEngine.rank_questions_for_mode(protocol, session, mode="vlm_best_question")

    assert len(questions) == 4
    assert "Is the target object blue?" not in [question.text for question in questions]
    assert questions[0].text == "Is it on the left side?"


def test_backend_fallback_questions_use_human_friendly_wording() -> None:
    session = _session(
        [
            _candidate(1, "yellow cup"),
            _candidate(2, "yellow cup"),
            _candidate(3, "blue cup"),
            _candidate(4, "red cup"),
        ]
    )
    ClarificationEngine.ensure_candidate_state(session)

    questions = ClarificationEngine._fallback_questions(session)
    texts = [question.text for question in questions]

    assert "Is it the yellow cup?" in texts
    assert any(text in texts for text in ["Is it on the left side?", "Is it on the right side?"])
    assert all("target object" not in text.lower() for text in texts)
    assert all("visible object" not in text.lower() for text in texts)


def test_backend_questions_remain_available_after_history_exhausts_new_options() -> None:
    session = _session([_candidate(1), _candidate(2), _candidate(3), _candidate(4)])
    ClarificationEngine.ensure_candidate_state(session)
    exhausted = ClarificationEngine._fallback_questions(session)
    session.question_history = [
        QuestionTurn(
            round_index=idx,
            question_id=question.id,
            text=question.text,
            answer="n",
            score_py=question.score_py,
            score_pn=question.score_pn,
            efe_score=question.efe_score,
        )
        for idx, question in enumerate(exhausted, start=1)
    ]

    questions = ClarificationEngine._fallback_questions(session)

    assert len(questions) == 4
    assert all(question.yes_candidate_ids for question in questions)
    assert all(question.no_candidate_ids for question in questions)


def test_answer_split_updates_state_and_can_resolve_without_vlm() -> None:
    session = _session([_candidate(1), _candidate(2)])
    ClarificationEngine.ensure_candidate_state(session)
    question = ClarificationEngine._question_from_parts(
        question_id=1,
        text="Is the target object on the left side?",
        plausible_ids=["cand_001", "cand_002"],
        yes_ids={"cand_001"},
        no_ids={"cand_002"},
    )
    assert question is not None
    session.current_question = question

    engine = ClarificationEngine(
        base_url="http://127.0.0.1:9/v1",
        model="dummy",
        system_prompt_path="services/a6000_web/prompts/system_prompt.txt",
    )
    engine.answer_current_question(session, "y")

    assert session.status == "resolved"
    assert session.resolved_target is not None
    assert session.resolved_target.candidate_id == "cand_001"
    assert session.plausible_candidate_ids == ["cand_001"]
    assert session.eliminated_candidate_ids == ["cand_002"]


def test_truncated_think_output_is_repaired_with_json_only_budget() -> None:
    session = _session(
        [
            _candidate(1, "green cup"),
            _candidate(2, "orange cup"),
            _candidate(3, "blue cup"),
            _candidate(4, "red cup"),
        ]
    )
    session.vlm_messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "start"}]
    protocol = _legal_protocol(session)
    engine = FakeChatEngine(
        [
            _response("<think>I need to reason through every candidate...", finish_reason="length"),
            _response(json.dumps(protocol), finish_reason="stop"),
        ],
        gen_max_tokens=256,
    )

    engine.advance_without_answer(session)

    assert session.status == "awaiting_answer"
    assert session.last_protocol_json == protocol
    assert len(session.current_questions) == 4
    assert engine.calls[1]["max_tokens"] >= 768
    assert "only one json object" in engine.calls[1]["messages"][-1]["content"].lower()
    assert "<think" not in session.vlm_messages[-1]["content"].lower()


def test_think_only_repair_failure_uses_backend_legal_protocol() -> None:
    session = _session(
        [
            _candidate(1, "green cup"),
            _candidate(2, "orange cup"),
            _candidate(3, "blue cup"),
            _candidate(4, "red cup"),
        ]
    )
    session.vlm_messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "start"}]
    engine = FakeChatEngine(
        [
            _response("<think>too long", finish_reason="length"),
            _response("<think>still too long", finish_reason="length"),
            _response("<think>still not json", finish_reason="length"),
        ],
        gen_max_tokens=256,
    )

    engine.advance_without_answer(session)

    assert session.status == "awaiting_answer"
    assert session.last_protocol_json is not None
    assert session.last_protocol_json["Head"] == "probose"
    assert "vlm_invalid_output" in session.last_protocol_json["Reason"]
    assert len(session.last_protocol_json["Question"]) == 4
    assert len(session.current_questions) == 4
    assert all(question.yes_candidate_ids for question in session.current_questions)
    assert all(question.no_candidate_ids for question in session.current_questions)
    assert "<think" not in session.vlm_messages[-1]["content"].lower()


def test_partial_single_candidate_not_enough_questions_uses_backend_decision() -> None:
    session = _session([_candidate(1, "yellow cup")])
    session.prompt_type = "partial"
    session.single_candidate_audit = True
    session.vlm_messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "start"}]
    bad_protocol = {
        "Head": "probose",
        "Task_ID": 1,
        "Round": 1,
        "Candidate_State": {"plausible": ["cand_001"], "eliminated": [], "last_removed": []},
        "Question": [],
    }
    engine = FakeChatEngine([_response(json.dumps(bad_protocol))])

    engine.advance_without_answer(session)

    assert session.status == "resolved"
    assert session.resolved_target is not None
    assert session.resolved_target.candidate_id == "cand_001"
    assert session.current_questions == []
    assert session.last_protocol_json is not None
    assert session.last_protocol_json["Head"] == "decision"
    assert "not_enough_legal_questions" in session.last_protocol_json["Reason"]


def test_partial_single_candidate_can_request_object_recall_without_questions() -> None:
    session = _session([_candidate(1, "red cup")])
    session.prompt_type = "partial"
    session.single_candidate_audit = True
    session.vlm_messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "start"}]
    recall_protocol = {
        "Head": "decision",
        "Task_ID": 1,
        "Round": 1,
        "Grasp": "no",
        "Need_Object_Recall": True,
        "Recall_Target": "cup",
        "Candidate_State": {"plausible": ["cand_001"], "eliminated": [], "last_removed": []},
        "Reason": "Another unboxed red cup may match.",
    }
    engine = FakeChatEngine([_response(json.dumps(recall_protocol))])

    engine.advance_without_answer(session)

    assert session.status == "needs_object_recall"
    assert session.resolved_target is None
    assert session.current_questions == []
    assert session.last_protocol_json == recall_protocol


def test_partial_expanded_candidates_can_be_pruned_before_backend_questions() -> None:
    session = _session(
        [
            _candidate(1, "cup"),
            _candidate(2, "cup"),
            _candidate(3, "cup"),
            _candidate(4, "cup"),
            _candidate(5, "cup"),
        ]
    )
    session.prompt_type = "partial"
    session.instruction = "Pick up the red cup."
    session.detection_prompt = "cup ."
    session.vlm_messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "start"}]
    protocol = {
        "Head": "probose",
        "Task_ID": 1,
        "Round": 1,
        "Candidate_State": {
            "plausible": ["cand_001", "cand_002"],
            "eliminated": ["cand_003", "cand_004", "cand_005"],
            "last_removed": ["cand_003", "cand_004", "cand_005"],
        },
        "Question": [],
    }
    engine = FakeChatEngine([_response(json.dumps(protocol))])

    engine.advance_without_answer(session)

    assert session.status == "awaiting_answer"
    assert session.plausible_candidate_ids == ["cand_001", "cand_002"]
    assert session.eliminated_candidate_ids == ["cand_003", "cand_004", "cand_005"]
    assert len(session.current_questions) == 4
    for question in session.current_questions:
        assert question.count["total"] == 2
        assert set(question.yes_candidate_ids) | set(question.no_candidate_ids) == {"cand_001", "cand_002"}
        assert not (set(question.yes_candidate_ids) | set(question.no_candidate_ids)) & {
            "cand_003",
            "cand_004",
            "cand_005",
        }


def test_partial_expanded_candidates_pruned_to_one_resolve_without_questions() -> None:
    session = _session([_candidate(1, "cup"), _candidate(2, "cup"), _candidate(3, "cup")])
    session.prompt_type = "partial"
    session.instruction = "Pick up the red cup."
    session.detection_prompt = "cup ."
    session.vlm_messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "start"}]
    protocol = {
        "Head": "probose",
        "Task_ID": 1,
        "Round": 1,
        "Candidate_State": {
            "plausible": ["cand_002"],
            "eliminated": ["cand_001", "cand_003"],
            "last_removed": ["cand_001", "cand_003"],
        },
        "Question": [],
    }
    engine = FakeChatEngine([_response(json.dumps(protocol))])

    engine.advance_without_answer(session)

    assert session.status == "resolved"
    assert session.resolved_target is not None
    assert session.resolved_target.candidate_id == "cand_002"
    assert session.current_questions == []
    assert session.last_protocol_json is not None
    assert session.last_protocol_json["Head"] == "decision"
