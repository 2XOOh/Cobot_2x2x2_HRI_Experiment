# voice_intent_interface.py
from __future__ import annotations

import json
import queue
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


DEFAULT_SYSTEM_PROMPT_PATH = Path(__file__).with_name("prompts") / "worker_intent_runtime.md"
DEFAULT_LLM_RESPONSE_LOG_PATH = Path(__file__).with_name("results") / "llm_response_log.jsonl"
LLM_ACTION_CONFIDENCE_THRESHOLD = 0.75
RULE_MIN_ANGLE_TOLERANCE_DEG = 1.0
SMALL_STEP_MAX_DELTA_DEG = 10.0
WORKER_ADJUSTMENT_CLARIFICATION_QUESTION = "높이를 유지할지, 올릴지, 내릴지 말씀해 주세요."


@dataclass
# LLM 의도 해석 결과를 main 루프에서 쓰기 쉬운 구조로 담는다.
class LlmAdjustmentDecision:
    # action: complete/adjust/reject/ask_clarification/unknown 등 LLM이 판단한 의도 분류다.
    action: str = "unknown"
    # direction: 조정 방향을 up/down/unclear 중 하나로 담는다.
    direction: str = "unclear"
    target_shoulder_angle_deg: float | None = None
    confidence: float = 0.0
    # reason: LLM이 판단 근거를 설명한 텍스트다.
    reason: str = ""
    # raw_text: 작업자 발화 원문 또는 LLM에 보낸 입력 문장이다.
    raw_text: str = ""
    # raw_response_text: LLM API가 돌려준 JSON 문자열 원문이다.
    raw_response_text: str = ""
    # source: 응답 출처를 llm/llm_error/rule_llm/manual 등으로 구분한다.
    source: str = "llm"
    # is_invalid: LLM이 명령을 무효 또는 부적절하다고 판단했는지 표시한다.
    is_invalid: bool = False
    # clarification_question: 추가 확인이 필요할 때 작업자에게 물어볼 문장이다.
    clarification_question: str = ""

    # LLM 의도 해석 결과를 dict 형태로 변환한다.
    def to_dict(self) -> dict[str, Any]:
        # CSV 기록이나 디버깅용으로 dataclass를 dict로 변환한다.
        return asdict(self)


# TTS 문장을 별도 큐와 thread로 순차 재생한다.
class QueuedTtsSpeaker:
    # TTS 요청을 큐에 쌓아 main loop를 막지 않고 순차적으로 말하게 한다.
    # TTS 큐와 worker thread 상태를 초기화한다.
    def __init__(self) -> None:
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # TTS worker thread를 필요할 때 시작한다.
    def start(self) -> None:
        # TTS worker thread가 없을 때만 새로 시작한다.
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    # 재생할 TTS 문장을 큐에 넣는다.
    def speak(self, text: str) -> None:
        # 말할 문장을 큐에 추가한다.
        if not text:
            return
        self.start()
        self._queue.put(text)

    # TTS worker thread에 종료 신호를 보낸다.
    def stop(self) -> None:
        # TTS worker thread에 종료 신호를 보낸다.
        self._queue.put(None)

    # 큐에 쌓인 TTS 문장이 모두 처리될 때까지 기다린다.
    def wait_until_done(self, timeout_sec: float | None = None) -> None:
        deadline = None if timeout_sec is None else time.time() + timeout_sec
        while self._queue.unfinished_tasks:
            if deadline is not None and time.time() >= deadline:
                return
            time.sleep(0.05)

    # 큐에서 문장을 꺼내 pyttsx3로 실제 음성을 재생한다.
    def _worker(self) -> None:
        # pyttsx3 엔진을 유지하면서 큐에 들어온 문장을 읽는다.
        while True:
            text = self._queue.get()
            engine = None
            try:
                if text is None:
                    return

                import pyttsx3

                engine = pyttsx3.init()
                engine.setProperty("rate", 160)
                engine.say(text)
                engine.runAndWait()
            except Exception as exc:
                print(f"[TTS ERROR] {exc}")
            finally:
                try:
                    if engine is not None:
                        engine.stop()
                except Exception:
                    pass
                self._queue.task_done()


# 백그라운드 마이크 인식을 수행하고 최신 발화만 보관한다.
class ContinuousSpeechRecognizer:
    # 마이크를 백그라운드에서 계속 듣고, 가장 최근 인식 문장을 보관한다.
    # 음성 인식 파라미터와 thread 상태를 초기화한다.
    def __init__(
        self,
        language: str = "ko-KR",
        energy_threshold: int = 300,
        dynamic_energy_threshold: bool = False,
        pause_threshold: float = 0.5,
        listen_timeout_sec: float = 1.0,
        phrase_time_limit_sec: float = 3.0,
        microphone_index: int | None = None,
        on_text: Callable[[str], None] | None = None,
    ) -> None:
        self.language = language
        self.energy_threshold = energy_threshold
        self.dynamic_energy_threshold = dynamic_energy_threshold
        self.pause_threshold = pause_threshold
        self.listen_timeout_sec = listen_timeout_sec
        self.phrase_time_limit_sec = phrase_time_limit_sec
        self.microphone_index = microphone_index
        self.on_text = on_text

        self._latest_text: str | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # 백그라운드 음성 인식 thread를 시작한다.
    def start(self) -> None:
        # 음성 인식 thread를 시작한다.
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # 백그라운드 음성 인식 thread에 정지 신호를 보낸다.
    def stop(self) -> None:
        # 음성 인식 thread 루프를 멈추도록 신호를 보낸다.
        self._stop_event.set()

    # 최신 음성 인식 결과를 반환하고 내부 버퍼를 비운다.
    def get_and_clear(self) -> str | None:
        # 가장 최근 음성 인식 결과를 가져오고 내부 버퍼를 비운다.
        with self._lock:
            text = self._latest_text
            self._latest_text = None
        return text

    # 새 음성 인식 결과를 저장하고 callback이 있으면 전달한다.
    def _set_latest(self, text: str) -> None:
        # 인식된 문장을 저장하고 필요하면 callback에 넘긴다.
        with self._lock:
            self._latest_text = text
        if self.on_text:
            self.on_text(text)

    # speech_recognition 루프에서 마이크 입력을 반복 처리한다.
    def _worker(self) -> None:
        # speech_recognition으로 짧은 발화를 반복해서 인식한다.
        import speech_recognition as sr

        recognizer = sr.Recognizer()
        recognizer.energy_threshold = self.energy_threshold
        recognizer.dynamic_energy_threshold = self.dynamic_energy_threshold
        recognizer.pause_threshold = self.pause_threshold

        microphone = sr.Microphone(device_index=self.microphone_index)
        with microphone as source:
            while not self._stop_event.is_set():
                try:
                    audio = recognizer.listen(
                        source,
                        timeout=self.listen_timeout_sec,
                        phrase_time_limit=self.phrase_time_limit_sec,
                    )
                    text = recognizer.recognize_google(audio, language=self.language)
                    self._set_latest(text)
                except sr.WaitTimeoutError:
                    continue
                except Exception:
                    continue


# 수동 완료 키 또는 LLM 해석으로 현재 task 완료 여부를 판단한다.
def is_task_completion_input(
    key: int,
    voice_text: str | None,
    llm_parser: "LlmIntentParser | None" = None,
    metadata: dict[str, Any] | None = None,
) -> bool:
    # 수동 완료 키를 제외한 음성 완료 판단은 LLM으로만 해석한다.
    if key == ord(" "):
        return True

    if not voice_text:
        return False
    if llm_parser is None:
        return False

    decision = llm_parser.parse(
        voice_text,
        context="task_completion",
        metadata=metadata,
    )
    if decision.source == "llm_error":
        print(f"[작업 완료 LLM 오류] {decision.reason}")
        return False
    return (
        decision.action == "complete"
        and decision.confidence >= LLM_ACTION_CONFIDENCE_THRESHOLD
        and not decision.is_invalid
    )


# Worker 주도 조건에서 수동/음성 응답을 해석해 조정 응답 dict를 만든다.
def parse_worker_adjustment_input(
    wait_start_time: float,
    key: int,
    voice_text: str | None,
    control_type: str,
    llm_parser: "LlmIntentParser | None" = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Worker 주도 조건에서 수동 키 입력과 LLM 응답을 한 곳에서 해석한다.
    elapsed_wait = time.time() - wait_start_time
    response = {
        "answered": False,
        "text": "",
        "action": "unknown",
        "direction": "unclear",
        "elapsed_wait": elapsed_wait,
        "target_shoulder_angle_deg": None,
        "latency": 0.0,
        "confidence": 0.0,
        "is_invalid": False,
        "reason": "",
        "source": "none",
        "clarification_question": "",
    }

    manual_action = _manual_adjustment_action(key)
    if manual_action:
        response.update(
            {
                "answered": manual_action in ("reject", "adjust"),
                "text": f"[Manual] {manual_action} adjustment",
                "action": manual_action,
                "source": "manual",
                "reason": "matched manual key",
                "confidence": 1.0,
                "clarification_question": (
                    WORKER_ADJUSTMENT_CLARIFICATION_QUESTION
                    if manual_action == "ask_clarification"
                    else ""
                ),
            }
        )
        return response

    if not voice_text:
        return response

    response["text"] = voice_text
    if control_type in ("LLM", "Rule"):
        if llm_parser is None:
            response.update(
                {
                    "is_invalid": True,
                    "reason": f"{control_type} worker response requires an LLM parser.",
                }
            )
            return response

        llm_start_time = time.time()
        llm_decision = llm_parser.parse(
            voice_text,
            context="adjustment_response",
            metadata=metadata,
        )
        response.update(
            {
                "action": llm_decision.action,
                "direction": llm_decision.direction,
                "target_shoulder_angle_deg": llm_decision.target_shoulder_angle_deg,
                "latency": time.time() - llm_start_time,
                "confidence": llm_decision.confidence,
                "is_invalid": llm_decision.is_invalid,
                "reason": llm_decision.reason,
                "source": llm_decision.source,
                "clarification_question": WORKER_ADJUSTMENT_CLARIFICATION_QUESTION,
            }
        )
        if llm_decision.source == "llm_error":
            return response

        if response["action"] == "ask_clarification" and not response["clarification_question"]:
            response["clarification_question"] = WORKER_ADJUSTMENT_CLARIFICATION_QUESTION

        retry_reason = _llm_worker_retry_reason(
            voice_text,
            response,
            metadata,
            validate_target=control_type == "LLM",
        )
        if (
            retry_reason
            and control_type == "LLM"
            and llm_decision.source == "llm"
            and _should_retry_llm_after_validation_failure(retry_reason)
        ):
            retry_metadata = _metadata_with_validation_feedback(
                metadata,
                retry_reason,
                response,
            )
            llm_decision = llm_parser.parse(
                voice_text,
                context="adjustment_response",
                metadata=retry_metadata,
            )
            response.update(
                {
                    "action": llm_decision.action,
                    "direction": llm_decision.direction,
                    "target_shoulder_angle_deg": llm_decision.target_shoulder_angle_deg,
                    "latency": time.time() - llm_start_time,
                    "confidence": llm_decision.confidence,
                    "is_invalid": llm_decision.is_invalid,
                    "reason": llm_decision.reason,
                    "source": llm_decision.source,
                    "clarification_question": WORKER_ADJUSTMENT_CLARIFICATION_QUESTION,
                }
            )
            if llm_decision.source == "llm_error":
                return response
            retry_reason = _llm_worker_retry_reason(
                voice_text,
                response,
                metadata,
                validate_target=True,
            )

        if (
            not retry_reason
            and control_type == "LLM"
            and response["action"] == "ask_clarification"
            and llm_decision.source == "llm"
            and _should_retry_llm_clarification(response, metadata)
        ):
            retry_metadata = _metadata_with_validation_feedback(
                metadata,
                "Clarification response should be semantically re-evaluated.",
                response,
            )
            llm_decision = llm_parser.parse(
                voice_text,
                context="adjustment_response",
                metadata=retry_metadata,
            )
            response.update(
                {
                    "action": llm_decision.action,
                    "direction": llm_decision.direction,
                    "target_shoulder_angle_deg": llm_decision.target_shoulder_angle_deg,
                    "latency": time.time() - llm_start_time,
                    "confidence": llm_decision.confidence,
                    "is_invalid": llm_decision.is_invalid,
                    "reason": llm_decision.reason,
                    "source": llm_decision.source,
                    "clarification_question": WORKER_ADJUSTMENT_CLARIFICATION_QUESTION,
                }
            )
            if llm_decision.source == "llm_error":
                return response
            retry_reason = _llm_worker_retry_reason(
                voice_text,
                response,
                metadata,
                validate_target=True,
            )

        if retry_reason:
            limit_action = _limit_action_from_retry_reason(retry_reason)
            response.update(
                {
                    "answered": False,
                    "action": limit_action or "unknown",
                    "target_shoulder_angle_deg": None,
                    "is_invalid": limit_action is None,
                    "reason": retry_reason,
                    "clarification_question": "",
                }
            )
            return response

        if control_type == "Rule":
            return _apply_rule_worker_policy(response, metadata)

        response["answered"] = response["action"] in ("approve", "reject", "adjust")
        return response

    response.update(
        {
            "is_invalid": True,
            "reason": f"Unsupported worker control type: {control_type}",
        }
    )
    return response


# Worker 응답 대기 중 입력된 수동 키를 조정 action으로 변환한다.
def _manual_adjustment_action(key: int) -> str | None:
    # Worker 응답 대기 중 Y/N 키를 approve/reject로 바꾼다.
    if key in (ord("y"), ord("Y")):
        return "ask_clarification"
    if key in (ord("n"), ord("N")):
        return "reject"
    return None


# LLM target이 허용되는 개인별 안전 어깨각 범위를 계산한다.
def _llm_safe_target_bounds(metadata: dict[str, Any]) -> tuple[float, float] | None:
    try:
        lower = float(metadata["effective_min_shoulder_deg"])
        pilot_max = float(metadata.get("pilot_functional_max_shoulder_deg", 80.0))
        robot_max = float(metadata.get("robot_max_reachable_shoulder_deg", max(lower, pilot_max)))
    except (KeyError, TypeError, ValueError):
        return None

    upper = min(robot_max, max(lower, pilot_max))
    if lower > upper:
        upper = lower
    return lower, upper


# Rule 조건에서 LLM target과 현재 대표각 차이로 방향만 추론한다.
def resolve_rule_worker_target_angle_by_policy(
    metadata: dict[str, Any] | None,
    direction: str,
) -> float | None:
    if not isinstance(metadata, dict):
        return None

    try:
        current_angle = float(metadata["cycle_representative_shoulder_angle_deg"])
    except (KeyError, TypeError, ValueError):
        return None

    if direction == "up":
        return max(0.0, min(180.0, current_angle + 1.0))
    if direction == "down":
        return max(0.0, min(180.0, current_angle - 1.0))
    return None


# Worker+Rule 응답을 최종 Rule 정책 응답으로 변환한다.
def _apply_rule_worker_policy(
    response: dict[str, Any],
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    llm_action = str(response.get("action", "unknown"))
    response["source"] = "rule_llm"

    if llm_action == "reject":
        response.update(
            {
                "answered": True,
                "target_shoulder_angle_deg": None,
                "reason": f"{response.get('reason', '')} Rule policy kept the current height.".strip(),
            }
        )
        return response

    direction = str(response.get("direction", "")).lower()
    if direction not in ("up", "down"):
        direction = _llm_adjustment_direction(response, metadata)
    if llm_action not in ("approve", "adjust") or direction is None:
        response.update(
            {
                "answered": False,
                "action": "unknown",
                "target_shoulder_angle_deg": None,
                "is_invalid": True,
                "reason": (
                    f"{response.get('reason', '')} "
                    "Rule policy could not determine an upward or downward direction."
                ).strip(),
            }
        )
        return response

    policy_reason = (
        "Rule policy will apply robot z +50mm."
        if direction == "up"
        else "Rule policy will apply robot z -50mm."
    )
    response.update(
        {
            "answered": True,
            "action": "adjust",
            "target_shoulder_angle_deg": None,
            "is_invalid": False,
            "reason": f"{response.get('reason', '')} {policy_reason}".strip(),
        }
    )
    return response


# LLM target과 현재 대표각을 비교해 up/down 방향을 추정한다.
def _llm_adjustment_direction(
    response: dict[str, Any],
    metadata: dict[str, Any] | None,
) -> str | None:
    if not isinstance(metadata, dict):
        return None
    try:
        current_angle = float(metadata["cycle_representative_shoulder_angle_deg"])
        llm_target = float(response["target_shoulder_angle_deg"])
    except (KeyError, TypeError, ValueError):
        return None

    if llm_target > current_angle:
        return "up"
    if llm_target < current_angle:
        return "down"
    try:
        effective_min = float(metadata["effective_min_shoulder_deg"])
    except (KeyError, TypeError, ValueError):
        return None
    if current_angle <= effective_min + RULE_MIN_ANGLE_TOLERANCE_DEG:
        return "down"
    return None


def _compact_utterance_text(voice_text: str) -> str:
    return re.sub(r"\s+", "", (voice_text or "").lower())


def _utterance_has_small_step_cue(voice_text: str) -> bool:
    text = _compact_utterance_text(voice_text)
    return bool(re.search(r"(조금|살짝|약간)", text))


def _should_retry_llm_after_validation_failure(retry_reason: str) -> bool:
    return retry_reason in {
        "Small-step target did not match the LLM target policy.",
    }


def _should_retry_llm_clarification(
    response: dict[str, Any],
    metadata: dict[str, Any] | None,
) -> bool:
    if str(response.get("action", "unknown")) != "ask_clarification":
        return False
    if not isinstance(metadata, dict):
        return False
    try:
        current_robot_angle = float(metadata["current_robot_shoulder_angle_deg"])
    except (KeyError, TypeError, ValueError):
        return False
    bounds = _llm_safe_target_bounds(metadata)
    if bounds is None:
        return False
    safe_lower, safe_upper = bounds
    tolerance_deg = RULE_MIN_ANGLE_TOLERANCE_DEG
    return (
        current_robot_angle <= safe_lower + tolerance_deg
        or current_robot_angle >= safe_upper - tolerance_deg
    )


def _metadata_with_validation_feedback(
    metadata: dict[str, Any] | None,
    retry_reason: str,
    response: dict[str, Any],
) -> dict[str, Any]:
    retry_metadata = dict(metadata or {})
    feedback: dict[str, Any] = {
        "reason": retry_reason,
        "previous_response": {
            "action": response.get("action"),
            "direction": response.get("direction"),
            "target_shoulder_angle_deg": response.get("target_shoulder_angle_deg"),
            "confidence": response.get("confidence"),
        },
        "instruction": "Return a corrected JSON object that satisfies this validation feedback.",
    }

    if retry_reason == "Small-step target did not match the LLM target policy.":
        direction = str(response.get("direction", "")).lower()
        policy_key = f"small_{direction}_target_deg"
        target_policy = retry_metadata.get("llm_target_policy", {})
        if isinstance(target_policy, dict) and policy_key in target_policy:
            feedback.update(
                {
                    "required_target_policy_key": policy_key,
                    "required_target_shoulder_angle_deg": target_policy[policy_key],
                    "instruction": (
                        "The utterance is a small-step adjustment. Use the required "
                        "llm_target_policy value exactly as target_shoulder_angle_deg."
                    ),
                }
            )

    if retry_reason == "Clarification response should be semantically re-evaluated.":
        feedback.update(
            {
                "instruction": (
                    "The previous response was ask_clarification. Re-evaluate the "
                    "whole utterance semantically. If it clearly asks to move higher "
                    "or lower, return action=adjust with the requested direction and "
                    "the matching llm_target_policy boundary/strength target. Keep "
                    "ask_clarification only if the utterance is truly directionless."
                ),
            }
        )

    retry_metadata["validation_feedback"] = feedback
    return retry_metadata


def _limit_action_from_retry_reason(retry_reason: str) -> str | None:
    if retry_reason in {
        "Downward intent was requested at or below the safe lower bound.",
        "Upward intent was requested at or above the safe upper bound.",
    }:
        return "limit_reached"
    return None


# LLM 응답이 적용 가능한지 confidence, 방향, 안전 범위 기준으로 검증한다.
def _llm_worker_retry_reason(
    voice_text: str,
    response: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    validate_target: bool = True,
) -> str:
    action = str(response.get("action", "unknown"))

    if action not in ("approve", "reject", "adjust"):
        return ""

    try:
        confidence = float(response.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    if confidence < LLM_ACTION_CONFIDENCE_THRESHOLD:
        return (
            f"LLM confidence {confidence:.2f} is below "
            f"{LLM_ACTION_CONFIDENCE_THRESHOLD:.2f}; asking worker to repeat."
        )

    if (
        validate_target
        and action in ("approve", "adjust")
        and response.get("target_shoulder_angle_deg") is None
    ):
        return "LLM selected height adjustment without a target angle; asking worker to repeat."

    if action in ("approve", "adjust"):
        direction = str(response.get("direction", "")).lower()
        if direction not in ("up", "down"):
            return "LLM selected height adjustment without a clear direction."

    if validate_target and action in ("approve", "adjust"):
        try:
            current_worker_angle = float((metadata or {})["cycle_representative_shoulder_angle_deg"])
            current_robot_angle = float((metadata or {})["current_robot_shoulder_angle_deg"])
            target_angle = float(response["target_shoulder_angle_deg"])
            effective_min = float((metadata or {})["effective_min_shoulder_deg"])
            robot_max = float((metadata or {})["robot_max_reachable_shoulder_deg"])
        except (KeyError, TypeError, ValueError):
            return "Robot-relative shoulder-angle metadata is missing or invalid."

        tolerance_deg = RULE_MIN_ANGLE_TOLERANCE_DEG
        condition = (metadata or {}).get("condition", {})
        llm_safe_policy_applies = isinstance(condition, dict) and condition.get("control") == "LLM"
        if llm_safe_policy_applies:
            bounds = _llm_safe_target_bounds(metadata or {})
            if bounds is None:
                return "LLM safe-range metadata is missing or invalid."
            safe_lower, safe_upper = bounds
            if target_angle < safe_lower - tolerance_deg or target_angle > safe_upper + tolerance_deg:
                return "LLM target angle is outside the safe 60-80 policy range."
            if direction == "down" and current_robot_angle <= safe_lower + tolerance_deg:
                return "Downward intent was requested at or below the safe lower bound."
            if direction == "up" and current_robot_angle >= safe_upper - tolerance_deg:
                return "Upward intent was requested at or above the safe upper bound."
            safe_baseline = max(safe_lower, min(safe_upper, current_robot_angle))
            if _utterance_has_small_step_cue(voice_text):
                target_policy = (metadata or {}).get("llm_target_policy", {})
                policy_key = f"small_{direction}_target_deg"
                try:
                    expected_target = float(target_policy[policy_key])
                except (KeyError, TypeError, ValueError):
                    expected_target = None
                if expected_target is not None:
                    if abs(target_angle - expected_target) > tolerance_deg:
                        return "Small-step target did not match the LLM target policy."
                elif abs(target_angle - safe_baseline) > SMALL_STEP_MAX_DELTA_DEG + tolerance_deg:
                    return "Small-step utterance produced an unexpectedly large target change."
            if not bool((metadata or {}).get("cycle_is_risky", False)):
                if direction == "up" and target_angle < safe_baseline - tolerance_deg:
                    return "Upward intent produced a target below the safe-range baseline."
                if direction == "down" and target_angle > safe_baseline + tolerance_deg:
                    return "Downward intent produced a target above the safe-range baseline."
        else:
            at_lower_limit = current_robot_angle <= effective_min + tolerance_deg
            at_upper_limit = current_robot_angle >= robot_max - tolerance_deg
            if direction == "up" and target_angle <= current_worker_angle:
                if not (at_upper_limit and target_angle >= current_worker_angle - tolerance_deg):
                    return "Upward intent produced a non-upward target angle."
            if direction == "down" and target_angle >= current_worker_angle:
                if not (at_lower_limit and target_angle <= current_worker_angle + tolerance_deg):
                    return "Downward intent produced a non-downward target angle."

    return ""


# OpenAI 호환 Chat Completion API로 발화를 구조화된 의도 결과로 파싱한다.
class LlmIntentParser:
    # LLM으로 작업자 의도와 상태를 해석해 조정 여부와 목표 어깨각을 구조화해서 받는다.
    # LLM 클라이언트, 모델, 프롬프트 경로를 초기화한다.
    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = "llama-3.3-70b-versatile",
        system_prompt_path: str | Path = DEFAULT_SYSTEM_PROMPT_PATH,
        temperature: float = 0.0,
        response_log_path: str | Path | None = DEFAULT_LLM_RESPONSE_LOG_PATH,
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.system_prompt_path = Path(system_prompt_path)
        self.response_log_path = Path(response_log_path) if response_log_path else None
        self.last_latency_s = 0.0
        self.last_response_record: dict[str, Any] | None = None

    # 발화와 metadata를 LLM에 보내고 구조화된 의도 결과를 반환한다.
    def parse(
        self,
        text: str | None,
        context: str = "any",
        metadata: dict[str, Any] | None = None,
    ) -> LlmAdjustmentDecision:
        # 발화와 context를 LLM에 보내고 LlmAdjustmentDecision으로 변환한다.
        started_at = time.time()
        self.last_latency_s = 0.0
        self.last_response_record = None
        raw_text = text or ""
        system_prompt = self.system_prompt_path.read_text(encoding="utf-8")
        user_content = self._build_user_content(raw_text, context, metadata)
        content = ""

        try:
            response = self._create_completion(system_prompt, user_content)
            content = response.choices[0].message.content or "{}"
            parsed = json.loads(content.strip())
            decision = self._result_from_json(parsed, raw_text, content)
            self.last_latency_s = time.time() - started_at
            self._write_response_log(
                context=context,
                raw_text=raw_text,
                metadata=metadata,
                user_content=user_content,
                raw_response_text=content,
                parsed_response=parsed,
                decision=decision,
            )
            return decision
        except Exception as exc:
            if "json_validate_failed" in str(exc):
                try:
                    repair_prompt = (
                        system_prompt
                        + "\nCRITICAL JSON REPAIR: Return final numeric literals only. "
                        + "Compute every addition or subtraction before writing JSON. "
                        + "Do not put arithmetic operators in any JSON value."
                    )
                    response = self._create_completion(repair_prompt, user_content)
                    content = response.choices[0].message.content or "{}"
                    parsed = json.loads(content.strip())
                    decision = self._result_from_json(parsed, raw_text, content)
                    self.last_latency_s = time.time() - started_at
                    self._write_response_log(
                        context=context,
                        raw_text=raw_text,
                        metadata=metadata,
                        user_content=user_content,
                        raw_response_text=content,
                        parsed_response=parsed,
                        decision=decision,
                        repaired=True,
                    )
                    return decision
                except Exception as retry_exc:
                    exc = retry_exc
            decision = LlmAdjustmentDecision(
                raw_text=raw_text,
                raw_response_text=content,
                source="llm_error",
                is_invalid=True,
                reason=f"LLM intent parsing failed: {exc}",
            )
            self.last_latency_s = time.time() - started_at
            self._write_response_log(
                context=context,
                raw_text=raw_text,
                metadata=metadata,
                user_content=user_content,
                raw_response_text=content,
                parsed_response=None,
                decision=decision,
                error=str(exc),
            )
            return decision

    # OpenAI 호환 chat completion 요청을 실행한다.
    def _create_completion(self, system_prompt: str, user_content: str):
        return self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=self.temperature,
            max_tokens=200,
            response_format={"type": "json_object"},
        )

    # LLM 원문 응답과 파싱된 decision을 jsonl 로그 파일에 저장한다.
    def _write_response_log(
        self,
        context: str,
        raw_text: str,
        metadata: dict[str, Any] | None,
        user_content: str,
        raw_response_text: str,
        parsed_response: dict[str, Any] | None,
        decision: LlmAdjustmentDecision,
        repaired: bool = False,
        error: str | None = None,
    ) -> None:
        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.model,
            "latency_s": self.last_latency_s,
            "context": context,
            "utterance": raw_text,
            "metadata": metadata or {},
            "user_content": user_content,
            "raw_response_text": raw_response_text,
            "parsed_response": parsed_response,
            "decision": decision.to_dict(),
            "repaired": repaired,
            "error": error,
        }
        self.last_response_record = record

        if self.response_log_path is None:
            return
        try:
            self.response_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.response_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f"[LLM LOG ERROR] {exc}")

    # LLM에 전달할 user message JSON을 구성한다.
    def _build_user_content(
        self,
        text: str,
        context: str,
        metadata: dict[str, Any] | None,
    ) -> str:
        # LLM user message를 JSON 문자열로 구성한다.
        payload = {
            "context": context,
            "utterance": text,
            "metadata": metadata or {},
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @staticmethod
    # LLM JSON 응답을 LlmAdjustmentDecision 객체로 변환한다.
    def _result_from_json(
        parsed: dict[str, Any],
        raw_text: str,
        raw_response_text: str = "",
    ) -> LlmAdjustmentDecision:
        # LLM이 반환한 JSON dict에서 action과 목표 어깨각만 main에서 쓰기 쉽게 꺼낸다.
        action = str(parsed.get("action", "unknown"))
        clarification_question = ""
        if action == "ask_clarification":
            clarification_question = WORKER_ADJUSTMENT_CLARIFICATION_QUESTION
        return LlmAdjustmentDecision(
            action=action,
            direction=str(parsed.get("direction", "unclear")).lower(),
            target_shoulder_angle_deg=LlmIntentParser._optional_float(
                parsed.get("target_shoulder_angle_deg")
                or parsed.get("target_angle_deg")
                or parsed.get("shoulder_angle_deg")
            ),
            confidence=float(parsed.get("confidence", 0.0) or 0.0),
            is_invalid=bool(parsed.get("is_invalid", False)),
            clarification_question=clarification_question,
            reason=str(parsed.get("reason", "")),
            raw_text=raw_text,
            raw_response_text=raw_response_text,
        )

    @staticmethod
    # None이나 문자열 숫자를 안전하게 float 또는 None으로 변환한다.
    def _optional_float(value: Any) -> float | None:
        # LLM이 숫자를 문자열로 줘도 목표 어깨각으로 쓸 수 있게 float로 변환한다.
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
